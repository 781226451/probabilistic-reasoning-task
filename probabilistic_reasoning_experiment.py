"""概率推理任务（PsychoPy 实现）。

任务逻辑：
1. 每个试次展示 6 个彩色图形（3 红 + 3 绿，顺序随机）。
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


class TrialResult(TypedDict):
    """单个试次的行为结果（用于 CSV 输出）。"""

    trial_num: int
    left_color: str
    right_color: str
    shapes_sequence: str
    colors_sequence: str
    left_weight: float
    right_weight: float
    weight_difference: float
    correct_response: str
    participant_response: str
    is_correct: bool
    reaction_time: float


class ShapeConfig(TypedDict):
    """单个 shape 的配置。"""

    image: str
    weight: float


# ==================== Experiment Parameters ====================
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
SHAPE_WEIGHTS_CONFIG_PATH: str = os.path.join(BASE_DIR, "assets", "shape_config.toml")
LOG_FILE_PATH: str = "data/experiment.log"

INITIAL_PROMPT_DURATION: int = 1500  # ms
STIMULUS_DURATION: int = 500         # ms
ISI_DURATION: int = 500              # ms
ITI_DURATION: int = 800              # ms
DECISION_TIMEOUT: int | None = 5000  # ms; 设为 None 表示不限时

N_STIMULI_PER_TRIAL: int = 6
CIRCLE_SIZE: int = 80
SHAPE_SIZE: int = 250
SCREEN_SIZE_PIX: list[int] = [3840, 2160]
SIDE_CIRCLE_X_OFFSET: int = 1400
SHAPE_IMAGE_DIR: str = "assets/shapes"
SHAPE_IMAGE_EXT: str = ".png"

# PsychoPy 颜色采用 [-1, 1] 范围的 RGB。
COLOR_RED: list[float] = [1, -1, -1]
COLOR_GREEN: list[float] = [-1, 1, -1]
COLOR_BLACK: list[float] = [-1, -1, -1]
COLOR_WHITE: list[float] = [1, 1, 1]
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


def check_escape(win: visual.Window) -> None:
    """全局 ESC 监听：一旦按下立即退出实验。"""
    if "escape" in event.getKeys(keyList=["escape"]):
        win.close()
        core.quit()


def safe_wait(win: visual.Window, duration_sec: float, check_interval: float = 0.01) -> None:
    """可中断等待函数。

    与 `core.wait` 不同，这里会在等待过程中不断轮询键盘，
    从而保证被试在任意阶段按 ESC 都能及时退出。
    """
    timer: core.Clock = core.Clock()
    while timer.getTime() < duration_sec:
        check_escape(win)
        remaining: float = duration_sec - timer.getTime()
        core.wait(min(check_interval, remaining))


def get_experiment_info() -> dict[str, Any]:
    """收集实验参数并转换成程序内部字段。"""
    exp_info: dict[str, Any] = {
        "被试编号": "",
        "试次数": 60,
        "是否反馈": ["否", "是"],
    }

    dlg = gui.DlgFromDict(dictionary=exp_info, title="概率推理任务", sortKeys=False)
    if not dlg.OK:
        core.quit()

    exp_info["timestamp"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_info["feedback_enabled"] = exp_info["是否反馈"] == "是"
    exp_info["participant_id"] = exp_info["被试编号"]
    exp_info["n_trials"] = exp_info["试次数"]
    return exp_info


def color_to_name(color: list[float]) -> str:
    """将 PsychoPy RGB 颜色值映射到颜色名。"""
    key = tuple(color)
    if key not in COLOR_NAME_MAP:
        raise ValueError(f"不支持的颜色值: {color}")
    return COLOR_NAME_MAP[key]


def create_shape_images(win: visual.Window, size: int = SHAPE_SIZE) -> dict[str, dict[str, visual.ImageStim]]:
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
            )
    return shape_images


def generate_trial(left_color: list[float], right_color: list[float]) -> Trial:
    """生成单个试次（刺激顺序 + 正确答案）。

    规则：
    - 图形序列长度固定为 `N_STIMULI_PER_TRIAL`。
    - 颜色固定 3 红 3 绿，再随机打乱。
    - 将每个图形权重累加到对应侧（由 left/right 颜色映射决定）。
    """
    if not SHAPE_WEIGHTS:
        raise RuntimeError("shape 权重尚未初始化，请先调用 initialize_shape_configs()")

    shape_names: list[str] = list(SHAPE_WEIGHTS.keys())
    shapes_sequence: list[str] = [random.choice(shape_names) for _ in range(N_STIMULI_PER_TRIAL)]

    colors: list[list[float]] = [COLOR_RED] * 3 + [COLOR_GREEN] * 3
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

    exp_info: dict[str, Any] = get_experiment_info()
    n_trials: int = int(exp_info["n_trials"])
    provide_feedback: bool = bool(exp_info["feedback_enabled"])

    if not os.path.exists("data"):
        os.makedirs("data")

    # 保存文件名包含被试编号与时间戳，避免覆盖历史数据。
    filename: str = f"data/sub-{exp_info['participant_id']}_{exp_info['timestamp']}"

    win: visual.Window | None = None
    try:
        win = visual.Window(
            size=SCREEN_SIZE_PIX,
            fullscr=True,
            color=COLOR_WHITE,
            units="pix",
            allowGUI=False,
        )

        shape_images: dict[str, dict[str, visual.ImageStim]] = create_shape_images(win)

        left_circle = visual.Circle(
            win,
            radius=CIRCLE_SIZE / 2,
            pos=[-SIDE_CIRCLE_X_OFFSET, 0],
            fillColor=COLOR_RED,
            lineColor=None,
        )
        center_circle = visual.Circle(win, radius=CIRCLE_SIZE / 2, pos=[0, 0], fillColor=COLOR_BLACK, lineColor=None)
        right_circle = visual.Circle(
            win,
            radius=CIRCLE_SIZE / 2,
            pos=[SIDE_CIRCLE_X_OFFSET, 0],
            fillColor=COLOR_GREEN,
            lineColor=None,
        )

        instruction_text = visual.TextStim(
            win,
            text=(
                "欢迎参加实验！\n"
                "每个试次会呈现一串不同颜色的图形图片。\n"
                "请按左右两侧颜色分别累加权重，并判断哪一侧更大。\n"
                "如果左侧总权重大于或等于右侧，请按左方向键（LEFT）。\n"
                "如果左侧总权重小于右侧，请按右方向键（RIGHT）。\n"
                "按空格键（SPACE）开始。"
            ),
            color=COLOR_BLACK,
            height=60,
            wrapWidth=1500,
        )

        decision_text = visual.TextStim(
            win,
            text="请作答\n\n左键（LEFT）：左侧总权重更高\n右键（RIGHT）：右侧总权重更高",
            color=COLOR_BLACK,
            height=35,
        )

        feedback_text = visual.TextStim(win, text="", color=COLOR_BLACK, height=40)

        instruction_text.draw()
        win.flip()
        start_key: list[str] | None = event.waitKeys(keyList=["space", "escape"])
        if start_key and "escape" in start_key:
            core.quit()

        # 采用平衡设计：一半试次左红右绿，另一半左绿右红。
        trials: list[Trial] = []
        for i in range(n_trials):
            if i < n_trials // 2:
                trial = generate_trial(COLOR_RED, COLOR_GREEN)
            else:
                trial = generate_trial(COLOR_GREEN, COLOR_RED)
            trials.append(trial)

        random.shuffle(trials)

        all_data: list[TrialResult] = []

        for trial_num, trial in enumerate(trials, 1):
            left_circle.fillColor = trial["left_color"]
            right_circle.fillColor = trial["right_color"]

            # 1) 初始提示阶段：先展示左右颜色映射与中心参考点。
            left_circle.draw()
            center_circle.draw()
            right_circle.draw()
            win.flip()
            safe_wait(win, INITIAL_PROMPT_DURATION / 1000.0)

            # 2) 序列呈现阶段：每个图形显示后接一个 ISI 空窗。
            for shape_name, color in zip(trial["shapes_sequence"], trial["colors_sequence"]):
                color_name = color_to_name(color)
                shape_image = shape_images[shape_name][color_name]

                left_circle.draw()
                center_circle.draw()
                right_circle.draw()
                shape_image.draw()
                win.flip()
                safe_wait(win, STIMULUS_DURATION / 1000.0)

                left_circle.draw()
                center_circle.draw()
                right_circle.draw()
                win.flip()
                safe_wait(win, ISI_DURATION / 1000.0)

            # 3) 决策阶段：等待左右键，支持超时与 ESC 中断。
            decision_text.draw()
            win.flip()

            decision_start_time: float = core.getTime()
            keys: list[str] | None = event.waitKeys(
                keyList=["left", "right", "escape"],
                maxWait=DECISION_TIMEOUT / 1000.0 if DECISION_TIMEOUT else None,
            )

            if keys is None:
                response: str = "timeout"
                rt: float = (DECISION_TIMEOUT or 0) / 1000.0
                is_correct: bool = False
            elif "escape" in keys:
                core.quit()
                return
            else:
                response = keys[0]
                rt = core.getTime() - decision_start_time
                is_correct = response == trial["correct_response"]

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
                safe_wait(win, 0.8)

            # 5) ITI：清屏等待，给下一试次留出间隔。
            win.flip()
            safe_wait(win, ITI_DURATION / 1000.0)

            trial_data: TrialResult = {
                "trial_num": trial_num,
                "left_color": "red" if trial["left_color"] == COLOR_RED else "green",
                "right_color": "red" if trial["right_color"] == COLOR_RED else "green",
                "shapes_sequence": ",".join(trial["shapes_sequence"]),
                "colors_sequence": ",".join(["red" if c == COLOR_RED else "green" for c in trial["colors_sequence"]]),
                "left_weight": trial["left_weight"],
                "right_weight": trial["right_weight"],
                "weight_difference": trial["left_weight"] - trial["right_weight"],
                "correct_response": trial["correct_response"],
                "participant_response": response,
                "is_correct": is_correct,
                "reaction_time": rt,
            }
            all_data.append(trial_data)

            print(f"第 {trial_num}/{n_trials} 试次完成")

        accuracy: float = (
            (sum(1 for d in all_data if d["is_correct"]) / len(all_data) * 100)
            if all_data
            else 0.0
        )

        end_text = visual.TextStim(
            win,
            text=f"实验结束！\n\n正确率：{accuracy:.1f}%\n\n感谢参与！",
            color=COLOR_BLACK,
            height=35,
        )
        end_text.draw()
        win.flip()
        safe_wait(win, 3)

        if all_data:
            # 使用 DictWriter 保持列顺序稳定，便于后续统计脚本解析。
            with open(f"{filename}.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_data[0].keys())
                writer.writeheader()
                rows_for_csv: list[dict[str, Any]] = [dict(row) for row in all_data]
                writer.writerows(rows_for_csv)

        print(f"数据已保存至：{filename}.csv")
        print(f"总体正确率：{accuracy:.1f}%")

        core.quit()
    finally:
        if win is not None:
            win.close()


if __name__ == "__main__":
    run_experiment()
