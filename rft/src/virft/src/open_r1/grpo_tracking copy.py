# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List

from datasets import load_dataset, load_from_disk, DatasetDict, Dataset, Features, Image as DatasetImage, Sequence
from transformers import Qwen2_5_VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen25VLGRPOTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json
import numpy as np

@dataclass
class TrackingGRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO tracking training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'tracking_iou', 'tracking_confidence', 'format'.
        max_pixels (`int`):
            Maximum number of pixels for the image.
        min_pixels (`int`):
            Minimum number of pixels for the image.
        frames_per_sequence (`int`):
            Number of frames to use per tracking sequence.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["tracking_iou", "tracking_confidence", "format"],
        metadata={"help": "List of reward functions. Possible values: 'tracking_iou', 'tracking_confidence', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    frames_per_sequence: Optional[int] = field(
        default=2, 
        metadata={"help": "Number of frames to use per tracking sequence"},
    )

def parse_bbox_string(bbox_str):
    """解析形如 '[x, y, w, h]' 或 '[x1, y1, x2, y2]' 的字符串"""
    try:
        coords_str = bbox_str.strip()[1:-1]
        coords = [int(c.strip()) for c in coords_str.split(',')]
        if len(coords) == 4:
            x, y, w, h = coords
            if w > 0 and h > 0:
                return [x, y, x + w, y + h]
            elif coords[2] > coords[0] and coords[3] > coords[1]:
                return coords
            else:
                return None
        return None
    except Exception:
        return None

def extract_bboxes_from_solution(solution_str: str) -> List[Optional[List[int]]]:
    """从 solution 字符串中提取所有帧的 BBox 列表 [x1, y1, x2, y2]"""
    bboxes = []
    pattern = re.compile(r"Frame\s*\d+:\s*(\[.*?\])", re.IGNORECASE)
    matches = pattern.findall(solution_str)
    for match in matches:
        bbox = parse_bbox_string(match)
        bboxes.append(bbox)
    if not bboxes and "<answer>" in solution_str:
        start_tag = "<answer>"
        end_tag = "</answer>"
        start_idx = solution_str.find(start_tag) + len(start_tag)
        end_idx = solution_str.find(end_tag)
        if end_idx == -1: end_idx = len(solution_str)
        content = solution_str[start_idx:end_idx].strip()
        single_bbox = parse_bbox_string(content)
        if single_bbox:
            bboxes.append(single_bbox)
    if not bboxes and solution_str.strip().startswith('[') and solution_str.strip().endswith(']'):
        try:
            potential_bboxes = json.loads(solution_str.strip().replace("'", '"'))
            if isinstance(potential_bboxes, list) and len(potential_bboxes) == 4:
                parsed = parse_bbox_string(solution_str.strip())
                if parsed: bboxes.append(parsed)
        except:
            pass
    return bboxes

def extract_bboxes_from_completion(completion_str: str) -> List[Optional[List[int]]]:
    """从模型 completion 字符串中提取所有帧的 BBox 列表 [x1, y1, x2, y2]"""
    start_tag = "<answer>"
    end_tag = "</answer>"
    content_to_parse = completion_str
    if start_tag in completion_str:
        start_idx = completion_str.find(start_tag) + len(start_tag)
        end_idx = completion_str.find(end_tag)
        if end_idx != -1:
            content_to_parse = completion_str[start_idx:end_idx].strip()
        else:
            content_to_parse = completion_str[start_idx:].strip()
    bboxes = []
    pattern = re.compile(r"Frame\s*\d+:\s*(\[.*?\])", re.IGNORECASE)
    matches = pattern.findall(content_to_parse)
    if matches:
        for match in matches:
            bbox = parse_bbox_string(match)
            bboxes.append(bbox)
        return bboxes
    single_bbox = parse_bbox_string(content_to_parse)
    if single_bbox:
        return [single_bbox]
    try:
        json_str = content_to_parse.replace("'", '"').replace("None", "null")
        parsed_json = json.loads(json_str)
        if isinstance(parsed_json, list):
            parsed_bboxes = []
            is_list_of_bboxes = True
            for item in parsed_json:
                if isinstance(item, list) and len(item) == 4:
                    parsed = parse_bbox_string(str(item))
                    parsed_bboxes.append(parsed)
                elif isinstance(item, str) and item.strip().lower() == "not visible":
                    parsed_bboxes.append(None)
                else:
                    is_list_of_bboxes = False
                    break
            if is_list_of_bboxes:
                return parsed_bboxes
    except Exception:
        pass
    return []

def calculate_tracking_iou(bbox1, bbox2):
    """
    计算两个边界框之间的IoU。
    如果两个框都为 None (表示不可见)，则返回 1.0。
    如果只有一个为 None，则返回 0.0。
    """
    if bbox1 is None and bbox2 is None:
        # Both predicted and GT are "not visible" -> Correct prediction
        return 1.0
    if bbox1 is None or bbox2 is None:
        # One is visible, the other is not -> Incorrect prediction (IoU is 0)
        return 0.0

    # --- Original IoU calculation logic for valid boxes ---
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Ensure coordinates are (x1, y1, x2, y2) where x1 < x2 and y1 < y2
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    if x1_2 > x2_2: x1_2, x2_2 = x2_2, x1_2
    if y1_2 > y2_2: y1_2, y2_2 = y2_2, y1_2

    # Check for invalid boxes (e.g., zero area after parsing)
    # Note: parse_bbox_string should already return None for [0,0,0,0]
    if x1 == x2 or y1 == y2 or x1_2 == x2_2 or y1_2 == y2_2:
        return 0.0 # Invalid box results in 0 IoU

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)

    # Calculate intersection area
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    # Calculate union area
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0 # Avoid division by zero or if areas are zero

    iou = inter_area / union_area
    return iou

def calculate_tracking_consistency(bbox_history):
    """计算追踪的一致性分数，基于运动的平滑度"""
    if len(bbox_history) < 2 or None in bbox_history:
        return 0.0
    
    centers = []
    for bbox in bbox_history:
        x1, y1, x2, y2 = bbox
        centers.append([(x1+x2)/2, (y1+y2)/2])
    
    displacements = []
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i-1][0]
        dy = centers[i][1] - centers[i-1][1]
        displacements.append(np.sqrt(dx*dx + dy*dy))
    
    if len(displacements) < 2:
        return 1.0
    
    std_dev = np.std(displacements)
    mean_disp = np.mean(displacements)
    
    if mean_disp == 0:
        return 1.0
    
    consistency = 1.0 - min(1.0, std_dev / max(1.0, mean_disp))
    return consistency

def calculate_trajectory_continuity(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    """
    计算轨迹连续性奖励。
    奖励连续的正确预测段，长度越长奖励越高（按长度平方和计算）。
    正确预测定义为：IoU >= threshold 或 预测和GT都为 None。
    分数通过总帧数的平方进行归一化。
    """
    num_frames = min(len(pred_bboxes), len(gt_bboxes))
    if num_frames == 0:
        return 0.0

    correct_predictions = []
    for i in range(num_frames):
        pred = pred_bboxes[i]
        gt = gt_bboxes[i]
        is_correct = False

        if pred is None and gt is None: # Both correctly identified as not visible
            is_correct = True
        elif pred is not None and gt is not None:
            # Use standard IoU calculation (None/None = 0) for threshold check
            # Re-calculate standard IoU here for clarity
            x1, y1, x2, y2 = pred
            x1_2, y1_2, x2_2, y2_2 = gt
            iou = 0.0
            if not (x1 > x2 or y1 > y2 or x1_2 > x2_2 or y1_2 > y2_2 or \
                    x1 == x2 or y1 == y2 or x1_2 == x2_2 or y1_2 == y2_2):
                xi1 = max(x1, x1_2); yi1 = max(y1, y1_2)
                xi2 = min(x2, x2_2); yi2 = min(y2, y2_2)
                inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
                area1 = (x2 - x1) * (y2 - y1); area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
                union_area = area1 + area2 - inter_area
                if union_area > 0: iou = inter_area / union_area

            if iou >= iou_threshold: # Correctly detected with sufficient IoU
                is_correct = True
        # else: False (Missed detection, False Positive, or Low IoU)

        correct_predictions.append(is_correct)

    if not any(correct_predictions):
        return 0.0 # No correct predictions, no continuity reward

    continuity_score = 0.0
    current_len = 0
    for is_correct in correct_predictions:
        if is_correct:
            current_len += 1
        else:
            if current_len > 0:
                continuity_score += current_len ** 2 # Add score for the ended segment
            current_len = 0
    if current_len > 0: # Add score for the last segment if it was correct
        continuity_score += current_len ** 2

    # Normalize the score by the square of the number of frames
    # This bounds the score between 0 and 1
    normalized_score = continuity_score / (num_frames ** 2) if num_frames > 0 else 0.0

    # Optional: Apply a non-linear function like sqrt to emphasize longer sequences less aggressively
    # normalized_score = np.sqrt(normalized_score)

    return normalized_score

def tracking_reward_iou(completions, solution, **kwargs):
    """基于多帧平均 IoU 的追踪奖励函数"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    for content, sol_str in zip(contents, solution):
        avg_iou = 0.0
        valid_frames = 0
        bboxes_pred = []
        bboxes_gt = []

        try:
            bboxes_pred = extract_bboxes_from_completion(content)
            bboxes_gt = extract_bboxes_from_solution(sol_str)

            num_frames = min(len(bboxes_pred), len(bboxes_gt))
            if num_frames > 0:
                total_iou = 0.0
                for i in range(num_frames):
                    pred_box = bboxes_pred[i]
                    gt_box = bboxes_gt[i]
                    if pred_box is not None and gt_box is not None:
                        iou = calculate_tracking_iou(pred_box, gt_box)
                        total_iou += iou
                        valid_frames += 1
                if valid_frames > 0:
                    avg_iou = total_iou / valid_frames
                elif any(b is not None for b in bboxes_gt):
                    avg_iou = -0.5
                elif not any(b is not None for b in bboxes_gt):
                    avg_iou = 0.1

            if len(bboxes_gt) > 0 and len(bboxes_pred) < len(bboxes_gt) * 0.5:
                avg_iou *= 0.8

        except Exception as e:
            avg_iou = 0.0

        rewards.append(avg_iou)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Multi-Frame Tracking IoU reward: {avg_iou} -------------\n")
                f.write(f"Model output: {content}\n")
                f.write(f"Reference: {sol_str}\n")
                f.write(f"Predicted bboxes: {bboxes_pred}\n")
                f.write(f"Ground truth bboxes: {bboxes_gt}\n")

    return rewards

def tracking_reward_confidence(completions, solution, **kwargs):
    """
    基于多帧置信度、一致性和轨迹连续性的追踪奖励函数。
    处理 "not visible" 情况 (GT=[0,0,0,0] 或 None)。
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # --- 定义各部分奖励的权重 ---
    W_IOU = 0.6           # IoU 权重
    W_CONSISTENCY = 0.1   # 运动一致性权重 (降低权重，因为它可能与轨迹连续性重叠)
    W_TRAJECTORY = 0.3    # 轨迹连续性权重
    IOU_THRESHOLD_FOR_TRAJECTORY = 0.5 # 用于判断轨迹连续性的 IoU 阈值

    for content, sol_str in zip(contents, solution):
        combined_reward = 0.0
        avg_iou = 0.0
        consistency_score = 0.0
        trajectory_score = 0.0
        valid_iou_frames = 0 # 用于计算平均 IoU 的有效帧数
        bboxes_pred = []
        bboxes_gt = []

        try:
            # 提取边界框列表 (解析器应将 [0,0,0,0] 转为 None)
            bboxes_pred = extract_bboxes_from_completion(content)
            bboxes_gt = extract_bboxes_from_solution(sol_str)

            # --- 1. 计算平均 IoU (使用修改后的 calculate_tracking_iou) ---
            num_frames = min(len(bboxes_pred), len(bboxes_gt))
            if num_frames > 0:
                total_iou = 0.0
                for i in range(num_frames):
                    pred_box = bboxes_pred[i]
                    gt_box = bboxes_gt[i]
                    # calculate_tracking_iou 现在处理 None/None 情况返回 1.0
                    iou = calculate_tracking_iou(pred_box, gt_box)
                    total_iou += iou
                    valid_iou_frames += 1 # 所有帧都参与 IoU 计算 (包括 None/None=1, None/Box=0)

                if valid_iou_frames > 0:
                    avg_iou = total_iou / valid_iou_frames
                # 如果 num_frames=0，avg_iou 保持 0.0

            # --- 2. 计算运动一致性 (只使用有效的预测框) ---
            valid_pred_bboxes = [b for b in bboxes_pred if b is not None]
            if len(valid_pred_bboxes) >= 2:
                # 注意: calculate_tracking_consistency 当前在遇到 None 时返回 0
                consistency_score = calculate_tracking_consistency(valid_pred_bboxes)
            elif len(valid_pred_bboxes) == 1:
                 # 单个有效预测，给一点基础分
                 consistency_score = 0.25
            elif len(valid_pred_bboxes) == 0 and not any(b is not None for b in bboxes_gt):
                 # 预测和 GT 都同意没有可见物体，视为一致
                 consistency_score = 1.0

            # --- 3. 计算轨迹连续性 ---
            trajectory_score = calculate_trajectory_continuity(
                bboxes_pred, bboxes_gt, IOU_THRESHOLD_FOR_TRAJECTORY
            )

            # --- 4. 结合 IoU, 一致性, 轨迹连续性 ---
            combined_reward = (W_IOU * avg_iou +
                               W_CONSISTENCY * consistency_score +
                               W_TRAJECTORY * trajectory_score)

            # 对帧数不匹配进行惩罚 (可选)
            if len(bboxes_gt) > 0 and len(bboxes_pred) < len(bboxes_gt):
                 # 如果预测的帧数少于 GT，可以按比例降低奖励
                 length_penalty = len(bboxes_pred) / len(bboxes_gt)
                 combined_reward *= (0.5 + 0.5 * length_penalty) # 例子：线性惩罚到 0.5

            # 确保奖励在合理范围内 (例如 0 到 1)
            combined_reward = max(0.0, min(1.0, combined_reward))

        except Exception as e:
            print(f"Error calculating confidence reward: {e}") # 打印错误用于调试
            combined_reward = 0.0

        rewards.append(combined_reward)

        # --- 调试日志 (可选) ---
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Multi-Frame Tracking Confidence reward: {combined_reward:.4f} -------------\n")
                f.write(f"Model output: {content}\n")
                f.write(f"Reference: {sol_str}\n")
                f.write(f"Predicted bboxes: {bboxes_pred}\n")
                f.write(f"Ground truth bboxes: {bboxes_gt}\n")
                f.write(f"Avg IoU (modified): {avg_iou:.4f}, Consistency: {consistency_score:.4f}, Trajectory: {trajectory_score:.4f}\n")
                f.write(f"Weights (IoU/Cons/Traj): {W_IOU}/{W_CONSISTENCY}/{W_TRAJECTORY}\n")

    return rewards

def format_reward(completions, **kwargs):
    """检查完成的格式是否正确的奖励函数 (适应多 BBox)"""
    pattern_answer_bbox = r"<answer>.*?\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\].*?</answer>"
    pattern_frame_bbox = r"Frame\s*\d+:\s*\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]"
    pattern_not_visible = r"(not visible|cannot see)"

    completion_contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content in completion_contents:
        has_answer_tag = "<answer>" in content and "</answer>" in content
        start_idx = content.find("<answer>") + len("<answer>")
        end_idx = content.find("</answer>")
        answer_content = content[start_idx:end_idx].strip() if has_answer_tag and end_idx > start_idx else ""

        is_format_ok = False
        if has_answer_tag and answer_content:
            if re.search(pattern_answer_bbox, content, re.DOTALL | re.IGNORECASE) or \
                re.search(pattern_frame_bbox, answer_content, re.IGNORECASE) or \
                re.search(pattern_not_visible, answer_content, re.IGNORECASE):
                is_format_ok = True
        elif not has_answer_tag and re.search(pattern_frame_bbox, content, re.IGNORECASE):
            is_format_ok = True

        rewards.append(1.0 if is_format_ok else 0.0)
    return rewards

reward_funcs_registry = {
    "tracking_iou": tracking_reward_iou,
    "tracking_confidence": tracking_reward_confidence,
    "format": format_reward,
}

def main(script_args, training_args, model_args):
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    raw_dataset = load_from_disk("/data1/lihaobo/tracking/rft/tracking_dataset")

    current_features = raw_dataset.features
    new_features_dict = current_features.copy()
    new_features_dict['image'] = Sequence(DatasetImage())
    new_features = Features(new_features_dict)

    # processed_dataset = raw_dataset.map(features=new_features, batched=True)
    processed_dataset = raw_dataset.cast(new_features)
    
    trainer_cls = Qwen25VLGRPOTrainer
    print("using trainer:", trainer_cls.__name__)

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=processed_dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

def main2():
    parser = TrlParser((TrackingGRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)