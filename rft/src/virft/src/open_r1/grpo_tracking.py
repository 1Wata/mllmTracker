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
from open_r1.trainer import Qwen25VLGRPOTrainer, Qwen25VLGRPOTrainerBackup
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json
import numpy as np
from open_r1.rewards.iou import calculate_iou
from open_r1.utils.utils import transform_bbox

# --- Start: Define transform_bbox if not available ---
def transform_bbox(bbox, original_size, resized_size, mode='resized_to_original'):
    """
    Transforms bounding box coordinates between original and resized image dimensions.

    Args:
        bbox (list[int or float]): Bounding box [x1, y1, x2, y2].
        original_size (tuple[int, int]): Original image size (height, width).
        resized_size (tuple[int, int]): Resized image size (height, width).
        mode (str): 'resized_to_original' or 'original_to_resized'.

    Returns:
        list[int]: Transformed bounding box [x1, y1, x2, y2], or None if input is invalid.
    """
    if not bbox or not original_size or not resized_size or len(bbox) != 4:
        return None # Invalid input

    orig_h, orig_w = original_size
    res_h, res_w = resized_size

    if orig_h == 0 or orig_w == 0 or res_h == 0 or res_w == 0:
        return None # Avoid division by zero

    x1, y1, x2, y2 = bbox

    if mode == 'resized_to_original':
        scale_w = orig_w / res_w
        scale_h = orig_h / res_h
        new_x1 = int(x1 * scale_w)
        new_y1 = int(y1 * scale_h)
        new_x2 = int(x2 * scale_w)
        new_y2 = int(y2 * scale_h)
    elif mode == 'original_to_resized':
        scale_w = res_w / orig_w
        scale_h = res_h / orig_h
        new_x1 = int(x1 * scale_w)
        new_y1 = int(y1 * scale_h)
        new_x2 = int(x2 * scale_w)
        new_y2 = int(y2 * scale_h)
    else:
        raise ValueError("Invalid mode specified. Use 'resized_to_original' or 'original_to_resized'.")

    # Clip coordinates to be within image boundaries (using original size for 'r_to_o', resized for 'o_to_r')
    target_w = orig_w if mode == 'resized_to_original' else res_w
    target_h = orig_h if mode == 'resized_to_original' else res_h

    new_x1 = max(0, min(new_x1, target_w - 1))
    new_y1 = max(0, min(new_y1, target_h - 1))
    new_x2 = max(0, min(new_x2, target_w - 1))
    new_y2 = max(0, min(new_y2, target_h - 1))

    # Ensure x1 <= x2 and y1 <= y2
    if new_x1 > new_x2: new_x1, new_x2 = new_x2, new_x1
    if new_y1 > new_y2: new_y1, new_y2 = new_y2, new_y1

    return [new_x1, new_y1, new_x2, new_y2]
# --- End: Define transform_bbox ---

@dataclass
class TrackingGRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO tracking training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'tracking_iou', 'format'.
        max_pixels (`int`):
            Maximum number of pixels for the image.
        min_pixels (`int`):
            Minimum number of pixels for the image.
        frames_per_sequence (`int`):
            Number of frames to use per tracking sequence.
        use_thinking (`bool`):
            Whether to use thinking process in model responses.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["tracking_iou", "format"],
        metadata={"help": "List of reward functions. Possible values: 'tracking_iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=102400,
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
    use_thinking: bool = field(
        default=False,
        metadata={"help": "Whether to use thinking process in model responses"},
    )

def extract_single_bbox(response):
    """
    从模型响应中提取单个边界框，自动适应多种响应格式
    
    Args:
        response (str): 模型的响应文本
        
    Returns:
        list: 提取的边界框坐标 [x1, y1, x2, y2]，如果无法提取则返回 None
    """
    # 尝试从不同格式中提取内容
    content_str = None
    
    # 检查是否有 thinking/answer 格式
    thinking_start = "<thinking>"
    thinking_end = "</thinking>"
    answer_start = "<answer>"
    answer_end = "</answer>"
    
    # 如果存在 thinking 和 answer 标签
    if thinking_start in response and answer_start in response:
        # 提取 answer 部分
        start_idx = response.find(answer_start) + len(answer_start)
        end_idx = response.find(answer_end) if answer_end in response else len(response)
        content_str = response[start_idx:end_idx].strip()
    
    # 如果只有 answer 标签
    elif answer_start in response:
        start_idx = response.find(answer_start) + len(answer_start)
        end_idx = response.find(answer_end) if answer_end in response else len(response)
        content_str = response[start_idx:end_idx].strip()
    
    # 如果没有标签，则尝试直接提取
    else:
        content_str = response.strip()
    
    # 如果没有内容可提取
    if not content_str:
        return None
    
    # 替换单引号为双引号以兼容JSON格式
    content_str = content_str.replace("'", '"')
    
    # 尝试解析为JSON格式
    # 方法1: 直接的坐标列表 [x1, y1, x2, y2]
    if content_str.startswith('[') and content_str.endswith(']'):
        # 尝试解析为JSON数组
        import json
        
        # 尝试解析整个内容
        bbox_data = None
        try:
            bbox_data = json.loads(content_str)
        except json.JSONDecodeError:
            # 如果解析失败，继续尝试下一种方法
            pass
        
        if bbox_data is not None:
            # 检查是直接的坐标列表还是带有Position键的对象列表
            if isinstance(bbox_data, list):
                if len(bbox_data) == 4 and all(isinstance(x, (int, float)) for x in bbox_data):
                    # 直接返回坐标列表 [x1, y1, x2, y2]
                    return bbox_data
                elif bbox_data and isinstance(bbox_data[0], dict) and 'Position' in bbox_data[0]:
                    # 返回第一个边界框的Position
                    return bbox_data[0]['Position']
    
    # 方法2: 尝试解析为字典形式 {'Position': [x1, y1, x2, y2]}
    if content_str.startswith('{') and content_str.endswith('}'):
        import json
        
        bbox_dict = None
        try:
            bbox_dict = json.loads(content_str)
        except json.JSONDecodeError:
            # 如果解析失败，继续尝试下一种方法
            pass
        
        if bbox_dict is not None and isinstance(bbox_dict, dict) and 'Position' in bbox_dict:
            return bbox_dict['Position']
    
    # 方法3: 使用正则表达式提取数字列表
    import re
    matches = re.findall(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', content_str)
    if matches:
        return [int(x) for x in matches[0]]
    
    return None

def extract_thinking_and_answer(response):
    """从模型响应中提取思考过程和最终答案"""
    thinking = None
    answer = None
    
    # 提取思考部分
    thinking_start_tag = "<thinking>"
    thinking_end_tag = "</thinking>"
    if thinking_start_tag in response and thinking_end_tag in response:
        thinking_start_idx = response.find(thinking_start_tag) + len(thinking_start_tag)
        thinking_end_idx = response.find(thinking_end_tag)
        if thinking_end_idx > thinking_start_idx:
            thinking = response[thinking_start_idx:thinking_end_idx].strip()
    
    # 提取答案部分
    answer_start_tag = "<answer>"
    answer_end_tag = "</answer>"
    if answer_start_tag in response:
        answer_start_idx = response.find(answer_start_tag) + len(answer_start_tag)
        answer_end_idx = response.find(answer_end_tag)
        if answer_end_idx == -1:
            answer_end_idx = len(response)
        answer = response[answer_start_idx:answer_end_idx].strip()
    
    return thinking, answer

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
    # Determine if completions are conversational or plain strings
    if isinstance(completions[0], list) and isinstance(completions[0][0], dict) and "content" in completions[0][0]:
        contents = [completion[0]["content"] for completion in completions]
    elif isinstance(completions[0], str):
        contents = completions
    else:
        raise ValueError("Unsupported completion format")

    # Get use_thinking from kwargs if passed, otherwise default (e.g., False)
    use_thinking = kwargs.get('use_thinking', False) # Default to False if not provided

    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    # Get size information from kwargs (expecting lists)
    original_sizes = kwargs.get('original_size', [None] * len(contents))
    resized_sizes = kwargs.get('resized_size', [None] * len(contents))

    for i, (content, sol) in enumerate(zip(contents, solution)):
        reward = 0.0
        original_size = original_sizes[i]
        resized_size = resized_sizes[i]

        # 从模型输出中提取边界框 (in resized coordinates)
        bbox_pred_resized = extract_single_bbox(content)
        # 从参考答案中提取边界框 (assumed in original coordinates)
        bbox_gt_orig = extract_single_bbox(sol)

        # Transform predicted bbox to original coordinates if sizes are available
        bbox_pred_orig = None
        if bbox_pred_resized and original_size and resized_size:
            bbox_pred_orig = transform_bbox(bbox_pred_resized, original_size, resized_size, 'resized_to_original')

        # Calculate IoU using boxes in the *same* (original) coordinate system
        if bbox_pred_orig and bbox_gt_orig:
            reward = calculate_iou(bbox_pred_orig, bbox_gt_orig)
        elif bbox_pred_resized and bbox_gt_orig and (not original_size or not resized_size):
             # Log if transformation couldn't happen due to missing size info
             print(f"Debug IoU: Calc skipped due to missing size info. Pred(res):{bbox_pred_resized}, GT(orig):{bbox_gt_orig}")
        elif bbox_pred_resized and bbox_gt_orig and not bbox_pred_orig:
             # Log if transformation failed but extraction worked
             print(f"Debug IoU: Calc skipped due to transform failure. Pred(res):{bbox_pred_resized}, GT(orig):{bbox_gt_orig}, Orig:{original_size}, Res:{resized_size}")

        rewards.append(reward)

        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Tracking IoU reward: {reward} -------------\n")
                f.write(f"Original Size: {original_size}, Resized Size: {resized_size}\n")
                f.write(f"Model output: {content}\n")
                f.write(f"Reference: {sol}\n")
                f.write(f"Predicted bbox (resized): {bbox_pred_resized}\n")
                f.write(f"Predicted bbox (original): {bbox_pred_orig}\n") # Log transformed box
                f.write(f"Ground truth bbox (original): {bbox_gt_orig}\n")

    return rewards



def format_reward(completions, solution, **kwargs):
    """检查完成的格式是否正确的奖励函数 (Unified interface)"""
    # Determine if completions are conversational or plain strings
    if isinstance(completions[0], list) and isinstance(completions[0][0], dict) and "content" in completions[0][0]:
        contents = [completion[0]["content"] for completion in completions]
    elif isinstance(completions[0], str):
        contents = completions
    else:
        raise ValueError("Unsupported completion format")

    # Pattern depends on whether thinking is used or not
    use_thinking = kwargs.get('use_thinking', False)
    if use_thinking:
        # Expect <thinking>...</thinking><answer>...</answer>
        pattern = r"<thinking>.*?</thinking>\s*<answer>\s*(\[.*?\]|\{.*?\})\s*</answer>"
    else:
        # Expect <answer>...</answer>
        pattern = r"<answer>\s*(\[.*?\]|\{.*?\})\s*</answer>"

    matches = [re.search(pattern, content, re.DOTALL) is not None for content in contents]
    return [1.0 if match else 0.0 for match in matches]

# 奖励函数注册表
reward_funcs_registry = {
    "tracking_iou": tracking_reward_iou,
    # "tracking_confidence": tracking_reward_confidence,
    "format": format_reward,
}

def main(script_args, training_args, model_args):
    # 获取奖励函数
    script_args.reward_funcs = ['tracking_iou', 'format']
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
    # trainer_cls = Qwen25VLGRPOTrainerBackup
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