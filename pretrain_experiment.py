"""范式实验前预训练任务（PsychoPy 实现）。"""

from __future__ import annotations

import csv
import os
import random
from datetime import datetime
from typing import Any, TypedDict

from psychopy import core, event, gui, visual

import probabilistic_reasoning_experiment as pre
from probabilistic_reasoning_experiment import (
    COLOR_BLACK,
    COLOR_GREEN,
    COLOR_RED,
    COLOR_WHITE,
    check_escape,
    create_shape_images,
    initialize_shape_configs,
    load_experiment_config,
    print_shape_weights,
    safe_wait,
    setup_experiment_logger,
)


class PretrainTrial(TypedDict):
    left_shape: str
    right_shape: str
    left_weight: float
    right_weight: float
    correct_response: str


def get_pretrain_info(profile_names: list[str]) -> dict[str, Any]:
    exp_info: dict[str, Any] = {
        "trial数": "12",
        "显示配置": profile_names,
    }

    while True:
        dlg = gui.DlgFromDict(dictionary=exp_info, title="预训练任务", sortKeys=False)
        if not dlg.OK:
            raise SystemExit("用户取消实验参数输入")

        try:
            trial_text = str(exp_info["trial数"]).strip()
            if not trial_text:
                raise ValueError("trial数不能为空")
            n_trials: int = int(trial_text)
            if n_trials <= 0:
                raise ValueError("trial数必须是大于 0 的整数")
        except (TypeError, ValueError) as exc:
            error_dlg = gui.Dlg(title="输入错误")
            error_dlg.addText(f"{exc}，请重新输入。")
            error_dlg.show()
            continue

        exp_info["timestamp"] = datetime.now().strftime("%Y%m%d%H%M%S")
        exp_info["n_trials"] = n_trials
        exp_info["display_profile_name"] = exp_info["显示配置"]
        return exp_info


def create_pretrain_folder(timestamp: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    folder = os.path.join(data_dir, f"pretrain_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    return folder


def generate_pretrain_trial() -> PretrainTrial:
    shape_names = list(pre.SHAPE_WEIGHTS.keys())
    if len(shape_names) < 2:
        raise ValueError("可用形状不足 2 个，无法进行预训练。")

    weighted_pairs: list[tuple[str, str]] = []
    for i in range(len(shape_names)):
        for j in range(i + 1, len(shape_names)):
            s1 = shape_names[i]
            s2 = shape_names[j]
            if pre.SHAPE_WEIGHTS[s1] != pre.SHAPE_WEIGHTS[s2]:
                weighted_pairs.append((s1, s2))

    if weighted_pairs:
        s1, s2 = random.choice(weighted_pairs)
    else:
        s1, s2 = random.sample(shape_names, 2)

    left_shape, right_shape = (s1, s2) if random.random() < 0.5 else (s2, s1)
    left_weight = pre.SHAPE_WEIGHTS[left_shape]
    right_weight = pre.SHAPE_WEIGHTS[right_shape]
    correct_response = "left" if left_weight >= right_weight else "right"

    return {
        "left_shape": left_shape,
        "right_shape": right_shape,
        "left_weight": left_weight,
        "right_weight": right_weight,
        "correct_response": correct_response,
    }


def run_pretrain() -> None:
    initialize_shape_configs()
    logger = setup_experiment_logger()
    print_shape_weights(logger)

    display_profiles, font_config, timing = load_experiment_config()
    profile_name_to_key = {v["name"]: k for k, v in display_profiles.items()}
    profile_names = list(profile_name_to_key.keys())

    exp_info = get_pretrain_info(profile_names)
    dp = display_profiles[profile_name_to_key[exp_info["display_profile_name"]]]
    n_trials = int(exp_info["n_trials"])
    pretrain_folder = create_pretrain_folder(exp_info["timestamp"])

    initial_prompt_sec = timing["initial_prompt_duration"] / 1000.0
    stimulus_sec = timing["stimulus_duration"] / 1000.0
    iti_sec = timing["iti_duration"] / 1000.0
    page_transition_sec = 0.2
    decision_timeout = timing["decision_timeout"]

    win: visual.Window | None = None
    rows: list[dict[str, Any]] = []

    try:
        win = visual.Window(
            size=dp["screen_size"],
            fullscr=True,
            color=COLOR_WHITE,
            units="pix",
            allowGUI=False,
        )

        shape_images = create_shape_images(win, size=dp["shape_size"])

        left_pos = [-dp["side_circle_x_offset"], 0]
        right_pos = [dp["side_circle_x_offset"], 0]
        left_circle = visual.Circle(
            win,
            radius=dp["circle_size"] / 2,
            pos=left_pos,
            fillColor=COLOR_RED,
            lineColor=None,
        )
        center_circle = visual.Circle(
            win,
            radius=dp["circle_size"] / 2,
            pos=[0, 0],
            fillColor=COLOR_BLACK,
            lineColor=None,
        )
        right_circle = visual.Circle(
            win,
            radius=dp["circle_size"] / 2,
            pos=right_pos,
            fillColor=COLOR_GREEN,
            lineColor=None,
        )

        instruction_text = visual.TextStim(
            win,
            text=(
                "预训练开始。\n"
                "每个 trial 会先显示三个圆形（左红/中黑/右绿）。\n"
                "随后左右会各出现一个绿色图形，请判断哪边更大。\n"
                "左箭头：左边大，右箭头：右边大。\n"
                "按空格键开始。"
            ),
            color=COLOR_BLACK,
            height=dp["instruction_text_height"],
            wrapWidth=dp["instruction_wrap_width"],
            font=font_config["name"],
            fontFiles=[font_config["file"]],
        )

        decision_text = visual.TextStim(
            win,
            text="哪边大？\n左箭头：左边大  右箭头：右边大",
            color=COLOR_BLACK,
            height=dp["decision_text_height"],
            font=font_config["name"],
            fontFiles=[font_config["file"]],
        )

        end_text = visual.TextStim(
            win,
            text="预训练结束，感谢参与。",
            color=COLOR_BLACK,
            height=dp["end_text_height"],
            font=font_config["name"],
            fontFiles=[font_config["file"]],
        )

        instruction_text.draw()
        win.flip()
        event.clearEvents(eventType="keyboard")
        start_key = event.waitKeys(keyList=["space", "escape"])
        if start_key and "escape" in start_key:
            return

        correct_count = 0

        for trial_index in range(1, n_trials + 1):
            check_escape()
            trial = generate_pretrain_trial()

            left_shape_stim = shape_images[trial["left_shape"]]["green"]
            right_shape_stim = shape_images[trial["right_shape"]]["green"]
            left_shape_stim.pos = left_pos
            right_shape_stim.pos = right_pos

            left_circle.draw()
            center_circle.draw()
            right_circle.draw()
            win.flip()
            safe_wait(initial_prompt_sec)

            left_circle.draw()
            center_circle.draw()
            right_circle.draw()
            left_shape_stim.draw()
            right_shape_stim.draw()
            win.flip()
            safe_wait(stimulus_sec)

            # 翻页到独立判断页（不显示圆和形状）
            win.flip()
            safe_wait(page_transition_sec)

            decision_text.text = "哪边大？\n左箭头：左边大  右箭头：右边大"
            decision_text.color = COLOR_BLACK
            decision_text.draw()
            win.flip()

            event.clearEvents(eventType="keyboard")
            rt_clock = core.Clock()
            keys = event.waitKeys(
                keyList=["left", "right", "escape"],
                maxWait=decision_timeout / 1000.0 if decision_timeout else None,
                timeStamped=rt_clock,  # type: ignore[arg-type]
            )

            if keys is None:
                response = "timeout"
                rt = (decision_timeout or 0) / 1000.0
                is_correct = False
            elif keys[0][0] == "escape":
                return
            else:
                response = keys[0][0]
                rt = keys[0][1]
                is_correct = response == trial["correct_response"]

            if is_correct:
                correct_count += 1

            answer_side = "左边" if trial["correct_response"] == "left" else "右边"
            response_side = "超时未作答"
            if response == "left":
                response_side = "左边"
            elif response == "right":
                response_side = "右边"

            decision_text.text = (
                "哪边大？\n"
                "左箭头：左边大  右箭头：右边大\n\n"
                f"你的回答：{response_side}\n"
                f"正确答案：{answer_side}\n"
                f"{'回答正确' if is_correct else '回答错误'}"
            )
            if not is_correct:
                left_weight_text = f"{trial['left_weight']:g}"
                right_weight_text = f"{trial['right_weight']:g}"
                decision_text.text += (
                    "\n\n本题形状权重：\n"
                    f"左侧 {trial['left_shape']}: {left_weight_text}\n"
                    f"右侧 {trial['right_shape']}: {right_weight_text}"
                )
            decision_text.text += "\n\n按空格键进入下一题"
            decision_text.color = [-1, 1, -1] if is_correct else [1, -1, -1]
            decision_text.draw()
            win.flip()
            event.clearEvents(eventType="keyboard")
            next_key = event.waitKeys(keyList=["space", "escape"])
            if next_key and "escape" in next_key:
                return

            rows.append(
                {
                    "trial": trial_index,
                    "left_shape": trial["left_shape"],
                    "right_shape": trial["right_shape"],
                    "left_weight": trial["left_weight"],
                    "right_weight": trial["right_weight"],
                    "correct_response": trial["correct_response"],
                    "response": response,
                    "is_correct": is_correct,
                    "rt_sec": f"{rt:.3f}",
                }
            )

            win.flip()
            safe_wait(iti_sec)

            logger.info(
                "Pretrain Trial %s | left=%s(%.3f) right=%s(%.3f) | resp=%s correct=%s rt=%.3f",
                trial_index,
                trial["left_shape"],
                trial["left_weight"],
                trial["right_shape"],
                trial["right_weight"],
                response,
                is_correct,
                rt,
            )

        accuracy = correct_count / n_trials * 100 if n_trials > 0 else 0.0

        end_text.text = f"预训练结束！\n\n正确率：{accuracy:.1f}%\n\n感谢参与。"
        end_text.draw()
        win.flip()
        safe_wait(3.0)

    finally:
        if rows:
            csv_path = os.path.join(pretrain_folder, "pretrain_trials.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "trial",
                        "left_shape",
                        "right_shape",
                        "left_weight",
                        "right_weight",
                        "correct_response",
                        "response",
                        "is_correct",
                        "rt_sec",
                    ],
                )
                writer.writeheader()
                writer.writerows(rows)
            logger.info("预训练结果已保存: %s", csv_path)

        if win is not None:
            win.close()
        core.quit()


if __name__ == "__main__":
    run_pretrain()
