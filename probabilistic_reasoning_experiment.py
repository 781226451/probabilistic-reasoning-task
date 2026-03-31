"""概率推理任务（PsychoPy 实现）。

任务逻辑：
1. 每个试次展示 `N_STIMULI_PER_TRIAL` 个彩色图形（红绿各半，顺序随机）。
2. 被试根据图形类型权重，分别累加左右两侧颜色对应的总权重。
3. 在决策阶段按左/右键选择较大的一侧。
"""

from __future__ import annotations

import csv
import logging
import os
import random
from datetime import datetime
from typing import Any, TypedDict

from psychopy import core, event, gui, visual
from pylsl import StreamInfo, StreamOutlet, cf_int8
import tomllib


class Trial(TypedDict):
    """单个试次的刺激配置与标准答案。"""

    left_color: list[float]
    right_color: list[float]
    shapes_sequence: list[str]
    colors_sequence: list[list[float]]
    left_weight: float
    right_weight: float
    correct_response: str


class ShapeConfig(TypedDict):
    """单个 shape 的配置。"""

    image: str
    weight: float


class EventLog(TypedDict):
    """事件日志记录。"""

    event_type: str
    timestamp: str
    content: str


class FontConfig(TypedDict):
    """字体配置。"""

    name: str
    file: str


class DisplayProfile(TypedDict):
    """单个显示配置文件的参数。"""

    name: str
    screen_size: list[int]
    shape_size: int
    circle_size: int
    side_circle_x_offset: int
    instruction_text_height: int
    instruction_wrap_width: int
    decision_text_height: int
    feedback_text_height: int
    end_text_height: int


class TimingConfig(TypedDict):
    """实验时序参数（单位：毫秒）。"""

    initial_prompt_duration: int
    stimulus_duration: int
    isi_duration: int
    iti_duration: int
    decision_timeout: int | None


# ==================== Experiment Parameters ====================
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
SHAPE_WEIGHTS_CONFIG_PATH: str = os.path.join(BASE_DIR, "assets", "shape_config.toml")
EXPERIMENT_CONFIG_PATH: str = os.path.join(BASE_DIR, "experiment_config.toml")
DATA_DIR: str = os.path.join(BASE_DIR, "data")
LOG_FILE_PATH: str = os.path.join(DATA_DIR, "experiment.log")

N_STIMULI_PER_TRIAL: int = 6
SHAPE_IMAGE_DIR: str = os.path.join(BASE_DIR, "assets", "shapes")
SHAPE_IMAGE_EXT: str = ".png"
LSL_STREAM_NAME: str = "ProbabilisticReasoning"
LSL_STREAM_TYPE: str = "Markers"
LSL_SOURCE_ID: str = "paradigm_seeg_marker"

# LSL marker 常量
LEFT_KEY_PRESSED = 16
RIGHT_KEY_PRESSED = 17

# PsychoPy 颜色采用 [-1, 1] 范围的 RGB。
COLOR_RED: list[float] = [1, -1, -1]
COLOR_GREEN: list[float] = [-1, 0.0039, -1]
COLOR_BLACK: list[float] = [-1, -1, -1]
COLOR_WHITE: list[float] = [1, 1, 1]
COLOR_GRAY: list[float] = [0, 0, 0]
COLOR_NAME_MAP: dict[tuple[float, ...], str] = {
    tuple(COLOR_RED): "red",
    tuple(COLOR_GREEN): "green",
}


def load_shape_configs(config_path: str = SHAPE_WEIGHTS_CONFIG_PATH) -> dict[str, ShapeConfig]:
    """从 TOML 文件加载 shape 配置（含 image 与 weight）并进行基本校验。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到权重配置文件: {config_path}")

    with open(config_path, "rb") as f:
        config_data: Any = tomllib.load(f)

    if not isinstance(config_data, dict):
        raise ValueError(f"权重配置格式错误（应为对象）: {config_path}")

    raw_shapes: Any = config_data.get("shapes")
    if not isinstance(raw_shapes, dict):
        raise ValueError(f"配置缺少 `shapes` 字段或字段类型错误: {config_path}")

    shape_configs: dict[str, ShapeConfig] = {}
    for shape_name, shape_config in raw_shapes.items():
        if not isinstance(shape_name, str) or not shape_name:
            raise ValueError("权重配置中的 shape 名必须是非空字符串")
        if not isinstance(shape_config, dict):
            raise ValueError(f"shape `{shape_name}` 的配置必须是对象")

        image_name: Any = shape_config.get("image")
        weight: Any = shape_config.get("weight")

        if not isinstance(image_name, str) or not image_name.strip():
            raise ValueError(f"shape `{shape_name}` 的 image 必须是非空字符串")
        if not isinstance(weight, (int, float)) or isinstance(weight, bool):
            raise ValueError(f"shape `{shape_name}` 的 weight 必须是数字")

        shape_configs[shape_name] = {
            "image": image_name.strip(),
            "weight": float(weight),
        }

    if not shape_configs:
        raise ValueError("权重配置不能为空")

    return shape_configs


SHAPE_CONFIGS: dict[str, ShapeConfig] = {}
SHAPE_WEIGHTS: dict[str, float] = {}


def initialize_shape_configs(config_path: str = SHAPE_WEIGHTS_CONFIG_PATH) -> None:
    """在运行时加载配置，避免模块导入阶段因文件问题直接崩溃。"""
    global SHAPE_CONFIGS, SHAPE_WEIGHTS
    SHAPE_CONFIGS = load_shape_configs(config_path)
    SHAPE_WEIGHTS = {shape_name: cfg["weight"] for shape_name, cfg in SHAPE_CONFIGS.items()}


def load_experiment_config(
    config_path: str = EXPERIMENT_CONFIG_PATH,
) -> tuple[dict[str, DisplayProfile], FontConfig, TimingConfig]:
    """从 TOML 文件加载实验配置（显示参数、字体、时序参数）。"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到实验配置文件: {config_path}")

    with open(config_path, "rb") as f:
        data: Any = tomllib.load(f)

    profiles: dict[str, Any] = data.get("profiles", {})
    if not profiles:
        raise ValueError("实验配置文件中没有有效的 profiles")

    raw_font: dict[str, Any] = data.get("fonts", {})
    if not raw_font:
        raise ValueError("实验配置文件中缺少 [fonts] 配置")

    font_config: FontConfig = {
        "name": raw_font["name"],
        "file": os.path.join(BASE_DIR, raw_font["file"]),
    }

    raw_timing: dict[str, Any] = data.get("timing", {})
    if not raw_timing:
        raise ValueError("实验配置文件中缺少 [timing] 配置")

    raw_timeout: Any = raw_timing.get("decision_timeout", 5000)
    timing: TimingConfig = {
        "initial_prompt_duration": int(raw_timing["initial_prompt_duration"]),
        "stimulus_duration": int(raw_timing["stimulus_duration"]),
        "isi_duration": int(raw_timing["isi_duration"]),
        "iti_duration": int(raw_timing["iti_duration"]),
        "decision_timeout": int(raw_timeout) if raw_timeout else None,
    }

    return profiles, font_config, timing  # type: ignore[return-value]


def setup_experiment_logger(log_path: str = LOG_FILE_PATH) -> logging.Logger:
    """创建实验日志器（写入文件）。"""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logger = logging.getLogger("experiment")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(file_handler)

    return logger


def print_shape_weights(logger: logging.Logger | None = None) -> None:
    """打印并记录配置中的 shape 名称与权重。"""
    if not SHAPE_CONFIGS:
        raise RuntimeError("shape 配置尚未初始化，请先调用 initialize_shape_configs()")

    message = f"已读取 shape 配置文件: {SHAPE_WEIGHTS_CONFIG_PATH}"
    print(message)
    if logger is not None:
        logger.info(message)

    for shape_name, shape_config in SHAPE_CONFIGS.items():
        line = f"- {shape_name}: {shape_config['weight']}"
        print(line)
        if logger is not None:
            logger.info(line)


def check_escape() -> None:
    """全局 ESC 监听：一旦按下立即退出实验。"""
    if "escape" in event.getKeys(keyList=["escape"]):
        raise SystemExit("用户按下 ESC 退出实验")


def safe_wait(duration_sec: float, check_interval: float = 0.01) -> None:
    """可中断等待函数。

    与 `core.wait` 不同，这里会在等待过程中不断轮询键盘，
    从而保证被试在任意阶段按 ESC 都能及时退出。
    """
    timer: core.Clock = core.Clock()
    while timer.getTime() < duration_sec:
        check_escape()
        remaining: float = duration_sec - timer.getTime()
        if remaining <= 0:
            break
        core.wait(min(check_interval, remaining))


def get_experiment_info(profile_names: list[str]) -> dict[str, Any]:
    """收集实验参数并转换成程序内部字段。"""
    exp_info: dict[str, Any] = {
        "被试编号": "",
        "trial数": 20,
        "是否反馈": ["否", "是"],
        "显示配置": profile_names,
    }

    while True:
        dlg = gui.DlgFromDict(dictionary=exp_info, title="概率推理任务", sortKeys=False)
        if not dlg.OK:
            raise SystemExit("用户取消实验参数输入")

        participant_id: str = str(exp_info["被试编号"]).strip()
        if not participant_id:
            error_dlg = gui.Dlg(title="输入错误")
            error_dlg.addText("被试编号不能为空，请重新输入。")
            error_dlg.show()
            continue

        try:
            n_trials: int = int(exp_info["trial数"])
            if n_trials <= 0:
                raise ValueError("trial数必须是大于 0 的整数")
        except (TypeError, ValueError) as exc:
            error_dlg = gui.Dlg(title="输入错误")
            error_dlg.addText(f"{exc}，请重新输入。")
            error_dlg.show()
            continue

        exp_info["timestamp"] = datetime.now().strftime("%Y%m%d%H%M%S")
        exp_info["feedback_enabled"] = exp_info["是否反馈"] == "是"
        exp_info["participant_id"] = participant_id
        exp_info["n_trials"] = n_trials
        exp_info["display_profile_name"] = exp_info["显示配置"]
        return exp_info

    raise RuntimeError("unreachable: get_experiment_info() exited input loop unexpectedly")


def color_to_name(color: list[float]) -> str:
    """将 PsychoPy RGB 颜色值映射到颜色名。"""
    key = tuple(color)
    if key not in COLOR_NAME_MAP:
        raise ValueError(f"不支持的颜色值: {color}")
    return COLOR_NAME_MAP[key]


def create_shape_images(win: visual.Window, size: int = 250) -> dict[str, dict[str, visual.ImageStim]]:
    """按配置加载图形的红/绿图片刺激。

    约定目录结构（优先）：
    - assets/shapes/<image>/<image>_red.png
    - assets/shapes/<image>/<image>_green.png
    同时兼容旧结构：
    - assets/shapes/<image>_red.png
    - assets/shapes/<image>_green.png
    """
    if not SHAPE_CONFIGS:
        raise RuntimeError("shape 配置尚未初始化，请先调用 initialize_shape_configs()")

    shape_images: dict[str, dict[str, visual.ImageStim]] = {}
    for shape_name, shape_config in SHAPE_CONFIGS.items():
        shape_images[shape_name] = {}
        image_base: str = shape_config["image"]
        for color_name in ("red", "green"):
            # 先按当前 assets 的分目录布局查找，再回退到旧版平铺布局。
            image_filename = f"{image_base}_{color_name}{SHAPE_IMAGE_EXT}"
            image_path = os.path.join(SHAPE_IMAGE_DIR, image_base, image_filename)
            if not os.path.exists(image_path):
                image_path = os.path.join(SHAPE_IMAGE_DIR, image_filename)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"未找到图形图片: {image_path}")
            shape_images[shape_name][color_name] = visual.ImageStim(
                win,
                image=image_path,
                size=(size, size),
                pos=(0, 0),
                color=(255,255,255),
                colorSpace='rgb255'
            )
    return shape_images


def generate_trial(left_color: list[float], right_color: list[float]) -> Trial:
    """生成单个试次（刺激顺序 + 正确答案）。

    规则：
    - 图形序列长度固定为 `N_STIMULI_PER_TRIAL`。
    - 颜色按 `N_STIMULI_PER_TRIAL` 动态生成红绿各半（要求为正偶数），再随机打乱。
    - 将每个图形权重累加到对应侧（由 left/right 颜色映射决定）。
    """
    if not SHAPE_WEIGHTS:
        raise RuntimeError("shape 权重尚未初始化，请先调用 initialize_shape_configs()")
    if N_STIMULI_PER_TRIAL <= 0 or N_STIMULI_PER_TRIAL % 2 != 0:
        raise ValueError("N_STIMULI_PER_TRIAL 必须是正偶数，以保证红绿数量平衡")

    shape_names: list[str] = list(SHAPE_WEIGHTS.keys())
    shapes_sequence: list[str] = [random.choice(shape_names) for _ in range(N_STIMULI_PER_TRIAL)]

    half_n: int = N_STIMULI_PER_TRIAL // 2
    colors: list[list[float]] = [COLOR_RED] * half_n + [COLOR_GREEN] * half_n
    random.shuffle(colors)

    left_weight: float = 0.0
    right_weight: float = 0.0

    for shape, color in zip(shapes_sequence, colors):
        weight: float = SHAPE_WEIGHTS[shape]
        if color == left_color:
            left_weight += weight
        else:
            right_weight += weight

    return {
        "left_color": left_color,
        "right_color": right_color,
        "shapes_sequence": shapes_sequence,
        "colors_sequence": colors,
        "left_weight": left_weight,
        "right_weight": right_weight,
        "correct_response": "left" if left_weight >= right_weight else "right",
    }


def create_experiment_folder(participant_id: str, timestamp: str, n_trials: int, feedback_enabled: bool, logger: logging.Logger) -> str:
    """在实验开始时创建独立文件夹并保存配置信息。

    文件夹命名格式：被试编号_年月日时分秒
    返回创建的文件夹路径
    """
    if not SHAPE_CONFIGS:
        raise RuntimeError("shape 配置尚未初始化，请先调用 initialize_shape_configs()")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 创建实验文件夹
    experiment_folder = os.path.join(DATA_DIR, f"{participant_id}_{timestamp}")
    os.makedirs(experiment_folder, exist_ok=True)

    # 创建config.txt文件
    config_filename = os.path.join(experiment_folder, "config.txt")
    with open(config_filename, "w", encoding="utf-8") as f:
        f.write("=" * 40 + "\n")
        f.write("实验配置信息\n")
        f.write("=" * 40 + "\n")
        f.write(f"被试编号: {participant_id}\n")
        f.write(f"实验开始时间: {timestamp}\n")
        f.write(f"trial个数: {n_trials}\n")
        f.write(f"是否进行反馈: {'是' if feedback_enabled else '否'}\n")
        f.write("\n")
        f.write("Shape权重配置:\n")
        f.write("-" * 40 + "\n")
        for shape_name, shape_config in SHAPE_CONFIGS.items():
            f.write(f"{shape_name}: {shape_config['weight']}\n")

    message = f"实验文件夹已创建：{experiment_folder}"
    print(message)
    logger.info(message)

    return experiment_folder


def run_experiment() -> None:
    """实验主流程。

    阶段：
    1. 参数采集与窗口初始化。
    2. 指导语显示与开始按键。
    3. 试次循环（提示 -> 序列呈现 -> 决策 -> 反馈 -> ITI）。
    4. 结果保存与结束页展示。
    """
    initialize_shape_configs()
    logger: logging.Logger = setup_experiment_logger()
    print_shape_weights(logger)

    display_profiles, font_config, timing = load_experiment_config()
    INITIAL_PROMPT_DURATION: int = timing["initial_prompt_duration"]
    STIMULUS_DURATION: int = timing["stimulus_duration"]
    ISI_DURATION: int = timing["isi_duration"]
    ITI_DURATION: int = timing["iti_duration"]
    DECISION_TIMEOUT: int | None = timing["decision_timeout"]
    profile_name_to_key: dict[str, str] = {v["name"]: k for k, v in display_profiles.items()}
    profile_names: list[str] = list(profile_name_to_key.keys())

    # 在程序起始阶段就打开 LSL outlet，确保不错过任何早期事件
    marker_info = StreamInfo(
        name=LSL_STREAM_NAME,
        type=LSL_STREAM_TYPE,
        channel_count=1,
        nominal_srate=0.0,
        channel_format=cf_int8,
        source_id=LSL_SOURCE_ID,
    )
    marker_outlet = StreamOutlet(marker_info)

    exp_info: dict[str, Any] = get_experiment_info(profile_names)
    dp: DisplayProfile = display_profiles[profile_name_to_key[exp_info["display_profile_name"]]]
    n_trials: int = int(exp_info["n_trials"])
    provide_feedback: bool = bool(exp_info["feedback_enabled"])

    # 创建实验文件夹并保存配置信息
    experiment_folder = create_experiment_folder(
        exp_info['participant_id'],
        exp_info['timestamp'],
        n_trials,
        provide_feedback,
        logger
    )

    # 记录实验信息到日志
    logger.info(f"实验开始 - 被试编号: {exp_info['participant_id']}, 时间戳: {exp_info['timestamp']}, trial数: {n_trials}, 是否反馈: {provide_feedback}")

    win: visual.Window | None = None
    event_logs: list[EventLog] = []  # 在try外部定义，确保finally块可以访问

    try:
        win = visual.Window(
            size=dp["screen_size"],
            fullscr=True,
            color=COLOR_GRAY,
            units="pix",
            allowGUI=False,
        )

        shape_images: dict[str, dict[str, visual.ImageStim]] = create_shape_images(win, size=dp["shape_size"])

        left_circle = visual.Circle(
            win,
            radius=dp["circle_size"] / 2,
            pos=[-dp["side_circle_x_offset"], 0],
            fillColor=COLOR_RED,
            lineColor=None,
        )
        center_circle = visual.Circle(win, radius=dp["circle_size"] / 2, pos=[0, 0], fillColor=COLOR_BLACK, lineColor=None)
        right_circle = visual.Circle(
            win,
            radius=dp["circle_size"] / 2,
            pos=[dp["side_circle_x_offset"], 0],
            fillColor=COLOR_GREEN,
            lineColor=None,
        )

        instruction_text = visual.TextStim(
            win,
            text=(
                "欢迎参加实验！\n"
                "每个试次会呈现一串不同颜色的图形图片\n"
                "请按左右两侧颜色分别累加权重，并判断哪一侧更大\n"
                "如果左侧总权重大于或等于右侧，请按左方向键（LEFT）\n"
                "如果左侧总权重小于右侧，请按右方向键（RIGHT）\n"
                "按空格键（SPACE）开始"
            ),
            color=COLOR_BLACK,
            height=dp["instruction_text_height"],
            wrapWidth=dp["instruction_wrap_width"],
            alignText="center",
            anchorHoriz="center",
            font=font_config["name"],
            fontFiles=[font_config["file"]],
        )

        decision_text = visual.TextStim(
            win,
            text="请作答\n\n左键（LEFT）：左侧总权重更高\n右键（RIGHT）：右侧总权重更高",
            color=COLOR_BLACK,
            height=dp["decision_text_height"],
            font=font_config["name"],
            fontFiles=[font_config["file"]],
        )

        feedback_text = visual.TextStim(
            win,
            text="",
            color=COLOR_BLACK,
            height=dp["feedback_text_height"],
            font=font_config["name"],
            fontFiles=[font_config["file"]],
        )

        instruction_text.draw()
        win.flip()
        event.clearEvents(eventType="keyboard")
        start_key: list[str] | None = event.waitKeys(keyList=["space", "escape"])
        if start_key and "escape" in start_key:
            return

        # 采用平衡设计：一半试次左红右绿，另一半左绿右红。
        trials: list[Trial] = []
        for i in range(n_trials):
            if i < n_trials // 2:
                trial = generate_trial(COLOR_RED, COLOR_GREEN)
            else:
                trial = generate_trial(COLOR_GREEN, COLOR_RED)
            trials.append(trial)

        random.shuffle(trials)

        correct_count: int = 0  # 统计正确次数

        # 记录实验开始事件
        event_logs.append({
            "event_type": "experiment_start",
            "timestamp": datetime.now().strftime("%Y%m%d%H%M%S.%f")[:-3],
            "content": f"实验开始，被试编号: {exp_info['participant_id']}, trial数: {n_trials}",
        })

        for trial_num, trial in enumerate(trials, 1):
            # 记录trial开始时间
            trial_start_time = datetime.now().strftime("%Y%m%d%H%M%S.%f")[:-3]  # 保留到毫秒
            logger.info(f"Trial {trial_num} 开始 - 时间: {trial_start_time}")

            left_circle.fillColor = trial["left_color"]
            right_circle.fillColor = trial["right_color"]

            left_color_name = "red" if trial["left_color"] == COLOR_RED else "green"
            right_color_name = "red" if trial["right_color"] == COLOR_RED else "green"

            event_logs.append({
                "event_type": "trial_start",
                "timestamp": trial_start_time,
                "content": f"Trial {trial_num} 开始, 左侧: {left_color_name}, 右侧: {right_color_name}",
            })

            # 1) 初始提示阶段：先展示左右颜色映射与中心参考点。
            left_circle.draw()
            center_circle.draw()
            right_circle.draw()
            win.flip()

            event_logs.append({
                "event_type": "initial_prompt",
                "timestamp": datetime.now().strftime("%Y%m%d%H%M%S.%f")[:-3],
                "content": f"Trial {trial_num} 初始提示显示 ({INITIAL_PROMPT_DURATION}ms)",
            })

            safe_wait(INITIAL_PROMPT_DURATION / 1000.0)

            # 2) 序列呈现阶段：每个图形显示后接一个 ISI 空窗。
            for stim_idx, (shape_name, color) in enumerate(zip(trial["shapes_sequence"], trial["colors_sequence"]), 1):
                color_name = color_to_name(color)
                shape_image = shape_images[shape_name][color_name]

                left_circle.draw()
                center_circle.draw()
                right_circle.draw()
                shape_image.draw()
                win.flip()

                event_logs.append({
                    "event_type": "stimulus_display",
                    "timestamp": datetime.now().strftime("%Y%m%d%H%M%S.%f")[:-3],
                    "content": f"Trial {trial_num} 刺激 {stim_idx}: {shape_name} ({color_name}), 权重: {SHAPE_WEIGHTS[shape_name]}",
                })

                safe_wait(STIMULUS_DURATION / 1000.0)

                left_circle.draw()
                center_circle.draw()
                right_circle.draw()
                win.flip()
                safe_wait(ISI_DURATION / 1000.0)

            # 3) 决策阶段：等待左右键，支持超时与 ESC 中断。
            decision_text.draw()
            win.flip()

            decision_start_timestamp = datetime.now().strftime("%Y%m%d%H%M%S.%f")[:-3]
            event_logs.append({
                "event_type": "decision_prompt",
                "timestamp": decision_start_timestamp,
                "content": f"Trial {trial_num} 决策提示显示",
            })

            # 清空此前阶段残留的键盘事件，避免提前按键污染本阶段 RT 与响应。
            event.clearEvents(eventType="keyboard")
            rt_clock: core.Clock = core.Clock()
            keys: list[tuple[str, float]] | None = event.waitKeys(
                keyList=["left", "right", "escape"],
                maxWait=DECISION_TIMEOUT / 1000.0 if DECISION_TIMEOUT else None,
                timeStamped=rt_clock,  # type: ignore[arg-type] # PsychoPy 实际支持 Clock
            )

            if keys is None:
                response: str = "timeout"
                rt: float = (DECISION_TIMEOUT or 0) / 1000.0
                is_correct: bool = False
            elif keys[0][0] == "escape":
                return
            else:
                response = keys[0][0]
                rt = keys[0][1]
                # 发送 LSL marker
                if response == "left":
                    marker_outlet.push_sample([LEFT_KEY_PRESSED])
                elif response == "right":
                    marker_outlet.push_sample([RIGHT_KEY_PRESSED])
                is_correct = response == trial["correct_response"]

            # 记录响应事件
            event_logs.append({
                "event_type": "response",
                "timestamp": datetime.now().strftime("%Y%m%d%H%M%S.%f")[:-3],
                "content": f"Trial {trial_num} 响应: {response}, 正确: {is_correct}, RT: {rt:.3f}s",
            })

            # 4) 反馈阶段（可选）：仅在非超时时显示对错。
            if provide_feedback and response != "timeout":
                if is_correct:
                    feedback_text.text = "正确"
                    feedback_text.color = [-1, 1, -1]
                else:
                    feedback_text.text = "错误"
                    feedback_text.color = [1, -1, -1]

                feedback_text.draw()
                win.flip()

                event_logs.append({
                    "event_type": "feedback",
                    "timestamp": datetime.now().strftime("%Y%m%d%H%M%S.%f")[:-3],
                    "content": f"Trial {trial_num} 反馈: {'正确' if is_correct else '错误'}",
                })

                safe_wait(0.8)

            # 5) ITI：清屏等待，给下一试次留出间隔。
            win.flip()
            safe_wait(ITI_DURATION / 1000.0)

            # 记录trial结束时间
            trial_end_time = datetime.now().strftime("%Y%m%d%H%M%S.%f")[:-3]  # 保留到毫秒
            logger.info(f"Trial {trial_num} 结束 - 时间: {trial_end_time}, 响应: {response}, 正确: {is_correct}")

            event_logs.append({
                "event_type": "trial_end",
                "timestamp": trial_end_time,
                "content": f"Trial {trial_num} 结束",
            })

            # 统计正确次数
            if is_correct:
                correct_count += 1

            print(f"第 {trial_num}/{n_trials} 试次完成")

        accuracy: float = (correct_count / n_trials * 100) if n_trials > 0 else 0.0

        # 记录实验结束事件
        event_logs.append({
            "event_type": "experiment_end",
            "timestamp": datetime.now().strftime("%Y%m%d%H%M%S.%f")[:-3],
            "content": f"实验结束，正确率: {accuracy:.1f}%",
        })

        end_text = visual.TextStim(
            win,
            text=f"实验结束！\n\n正确率：{accuracy:.1f}%\n\n感谢参与！",
            color=COLOR_BLACK,
            height=dp["end_text_height"],
            font=font_config["name"],
            fontFiles=[font_config["file"]],
        )
        end_text.draw()
        win.flip()
        safe_wait(3)

        print(f"总体正确率：{accuracy:.1f}%")
    finally:
        # 保存事件日志到CSV文件（在finally块中确保总是执行）
        if event_logs:
            event_log_filename = os.path.join(experiment_folder, "events.csv")
            try:
                with open(event_log_filename, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=["event_type", "timestamp", "content"])
                    writer.writeheader()
                    rows_for_csv: list[dict[str, Any]] = [dict(row) for row in event_logs]
                    writer.writerows(rows_for_csv)
                print(f"事件日志已保存至：{event_log_filename}")
                logger.info(f"事件日志已保存至：{event_log_filename}")
            except OSError as exc:
                error_message = f"事件日志保存失败: {exc}"
                print(error_message)
                logger.error(error_message)

        if win is not None:
            win.close()
        core.quit()

if __name__ == "__main__":
    run_experiment()
