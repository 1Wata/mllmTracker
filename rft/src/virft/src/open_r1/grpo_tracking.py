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
# from open_r1.trainer import Qwen25VLGRPOTrainer, Qwen25VLGRPOVLLMTrainer
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

def extract_single_bbox(response):
    """从模型响应中提取单个边界框"""
    start_tag = "<answer>"
    end_tag = "</answer>"
    
    if start_tag not in response:
        return None
    
    # 提取标签之间的内容
    start_idx = response.find(start_tag) + len(start_tag)
    end_idx = response.find(end_tag)
    
    if end_idx == -1:
        end_idx = len(response)
    
    content_str = response[start_idx:end_idx].strip()
    
    # 尝试解析为JSON
    try:
        # 替换单引号为双引号以兼容JSON格式
        content_str = content_str.replace("'", '"')
        # 处理可能的格式问题
        content_str = content_str.replace("[", "[").replace("]", "]")
        
        # 支持两种可能的格式：直接的坐标列表或者带有Position键的字典
        if content_str.startswith('[') and content_str.endswith(']'):
            try:
                bbox_data = json.loads(content_str)
                if isinstance(bbox_data, list):
                    if len(bbox_data) == 4 and all(isinstance(x, (int, float)) for x in bbox_data):
                        # 直接返回坐标列表 [x1, y1, x2, y2]
                        return bbox_data
                    elif isinstance(bbox_data[0], dict) and 'Position' in bbox_data[0]:
                        # 返回第一个边界框的Position
                        return bbox_data[0]['Position']
            except:
                # 尝试其他格式解析方法
                pass
                
        # 尝试解析为字典形式 {'Position': [x1, y1, x2, y2]}
        try:
            bbox_dict = json.loads(content_str)
            if isinstance(bbox_dict, dict) and 'Position' in bbox_dict:
                return bbox_dict['Position']
        except:
            pass
            
    except Exception as e:
        pass
        
    # 如果所有解析都失败，尝试使用正则表达式提取数字列表
    try:
        import re
        # 查找形如 [x, y, x, y] 的模式
        matches = re.findall(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', content_str)
        if matches:
            # 使用第一个匹配的结果
            return [int(x) for x in matches[0]]
    except:
        pass
        
    return None

def calculate_tracking_iou(bbox1, bbox2):
    """计算两个边界框之间的IoU"""
    if bbox1 is None or bbox2 is None:
        return 0.0
        
    x1, y1, x2, y2 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # 确保坐标有效
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    if x1_2 > x2_2: x1_2, x2_2 = x2_2, x1_2
    if y1_2 > y2_2: y1_2, y2_2 = y2_2, y1_2

    # 计算交集
    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    intersection_area = (xi2 - xi1) * (yi2 - yi1)
    
    # 计算并集
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    if area1 <= 0 or area2 <= 0:
        return 0.0

    union_area = area1 + area2 - intersection_area
    
    # 计算IoU
    iou = intersection_area / union_area
    return iou

def calculate_tracking_consistency(bbox_history):
    """计算追踪的一致性分数，基于运动的平滑度"""
    if len(bbox_history) < 2 or None in bbox_history:
        return 0.0
    
    # 计算连续帧之间的中心点位移
    centers = []
    for bbox in bbox_history:
        x1, y1, x2, y2 = bbox
        centers.append([(x1+x2)/2, (y1+y2)/2])
    
    displacements = []
    for i in range(1, len(centers)):
        dx = centers[i][0] - centers[i-1][0]
        dy = centers[i][1] - centers[i-1][1]
        displacements.append(np.sqrt(dx*dx + dy*dy))
    
    # 如果只有一个位移，无法计算一致性
    if len(displacements) < 2:
        return 1.0
    
    # 计算位移变化的标准差，标准差越小表示运动越平滑
    std_dev = np.std(displacements)
    mean_disp = np.mean(displacements)
    
    # 归一化标准差，使其成为0-1之间的分数
    if mean_disp == 0:
        return 1.0  # 没有移动，认为是完美追踪
    
    consistency = 1.0 - min(1.0, std_dev / max(1.0, mean_disp))
    return consistency

def tracking_reward_iou(completions, solution, **kwargs):
    """基于IoU的追踪奖励函数"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        
        try:
            # 从模型输出中提取边界框
            bbox_pred = extract_single_bbox(content)
            # 从参考答案中提取边界框
            bbox_gt = extract_single_bbox(sol)
            
            if bbox_pred and bbox_gt:
                # 计算IoU作为奖励
                reward = calculate_tracking_iou(bbox_pred, bbox_gt)
            
        except Exception as e:
            pass
        
        rewards.append(reward)
        

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Tracking IoU reward: {reward} -------------\n")
                f.write(f"Model output: {content}\n")
                f.write(f"Reference: {sol}\n")
                f.write(f"Predicted bbox: {bbox_pred}\n")
                f.write(f"Ground truth bbox: {bbox_gt}\n")
                
    return rewards

def tracking_reward_confidence(completions, solution, tracking_history=None, **kwargs):
    """基于置信度和一致性的追踪奖励函数"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    
    for content, sol in zip(contents, solution):
        reward = 0.0
        

        bbox_pred = extract_single_bbox(content)
        bbox_gt = extract_single_bbox(sol)
        
        if bbox_pred and bbox_gt:
            # 基本IoU奖励
            iou_score = calculate_tracking_iou(bbox_pred, bbox_gt)
            
            # 如果有追踪历史，计算一致性奖励
            consistency_score = 0.0
            if tracking_history and isinstance(tracking_history, list) and len(tracking_history) > 0:
                # 将当前预测添加到历史记录中进行一致性评估
                history_with_current = tracking_history + [bbox_pred]
                consistency_score = calculate_tracking_consistency(history_with_current)
            
            # 结合IoU和一致性的奖励
            reward = 0.7 * iou_score + 0.3 * consistency_score

            
        rewards.append(reward)
        
        # 调试日志
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Tracking confidence reward: {reward} -------------\n")
                f.write(f"Model output: {content}\n")
                f.write(f"Reference: {sol}\n")
                f.write(f"Predicted bbox: {bbox_pred}\n")
                f.write(f"Ground truth bbox: {bbox_gt}\n")
                if tracking_history:
                    f.write(f"Tracking history: {tracking_history}\n")
                
    return rewards

def format_reward(completions, **kwargs):
    """检查完成的格式是否正确的奖励函数"""
    pattern = r"<answer>\s*(\[.*?\]|\{.*?\})\s*</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

# 奖励函数注册表
reward_funcs_registry = {
    "tracking_iou": tracking_reward_iou,
    "tracking_confidence": tracking_reward_confidence,
    "format": format_reward,
}



TRACKING_SYSTEM_PROMPT = (
    "You are a professional visual object tracking assistant. Your task is to track specified target objects in a video sequence. "
    "The user will provide an initial frame with the target's bounding box, then you need to find the target's new position in subsequent frames. "
    "Please directly return the target's bounding box coordinates in the format [x1, y1, x2, y2], where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate. "
    "Your answer should be wrapped in <answer>[x1, y1, x2, y2]</answer> tags."
)



def main(script_args, training_args, model_args):
    # 获取奖励函数
    script_args.reward_funcs = ['tracking_iou', 'tracking_confidence', 'format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # from data_converter import create_dataset, load_image
    # processed_dataset = create_dataset(script_args.dataset_name, os.path.abspath(os.path.dirname(script_args.dataset_name)))
    # processed_dataset = dataset.map(load_image)

    # 处理数据集
    # processed_dataset = dataset.map(make_tracking_conversation)

    from datasets import load_from_disk
    raw_dataset = load_from_disk(script_args.dataset_name)

    current_features = raw_dataset.features
    new_features_dict = current_features.copy()
    new_features_dict['image'] = Sequence(DatasetImage())
    new_features = Features(new_features_dict)

    # processed_dataset = raw_dataset.map(features=new_features, batched=True)
    processed_dataset = raw_dataset.cast(new_features)
    # 确定使用哪个Trainer类
    # trainer_cls = Qwen25VLGRPOVLLMTrainer if training_args.use_vllm else Qwen25VLGRPOTrainer
    trainer_cls = Qwen25VLGRPOTrainer
    print("using trainer:", trainer_cls.__name__)

    
    
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        # train_dataset=processed_dataset[script_args.dataset_train_split],
        train_dataset=processed_dataset,
        # eval_dataset=processed_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # 训练模型
    trainer.train()


    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
# def main2():
    parser = TrlParser((TrackingGRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)