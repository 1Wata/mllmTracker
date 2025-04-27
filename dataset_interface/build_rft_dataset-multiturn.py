import os
from PIL import Image
import json
from tqdm import tqdm
import random
import numpy as np
import torch
from datasets import Dataset
from utils.utils import normalize_bbox_xyhw

def build_multi_turn_tracking_dataset(pytorch_dataset, output_dir="multi_turn_tracking_dataset"):
    """
    Converts a sequential PyTorch Dataset to a Huggingface Dataset in multi-turn conversation format
    for single object tracking.
    
    Args:
        pytorch_dataset: PyTorch Dataset instance with sequential tracking data
        output_dir: Directory to save the dataset
    """
    data_len = len(pytorch_dataset)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Lists to store dataset items
    all_samples = []
    
    skipped_count = 0
    for i in tqdm(range(data_len), desc="Building Multi-turn Dataset"):
        sample = pytorch_dataset[i]
        
        frame_paths = sample.get("images")
        annotations_dict = sample.get("anno")
        dataset_name = sample.get("dataset_name", "unknown")
        exp_str = sample.get("exp_str", "").strip("\n")
        
        num_frames = len(frame_paths)
        if num_frames < 2:
            skipped_count += 1
            continue
            
        # Process frames and annotations
        image_dimensions = []
        valid_list = annotations_dict.get('valid', [True] * num_frames)
        
        # Extract image dimensions
        for frame_path in frame_paths:
            img = Image.open(frame_path)
            image_dimensions.append(img.size)
        
        # Create annotation list
        processed_annotations = []
        for frame_idx in range(num_frames):
            current_bbox = annotations_dict['bbox'][frame_idx].tolist()
            current_visible = annotations_dict['visible'][frame_idx]
            current_valid = valid_list[frame_idx]
            
            current_anno = {
                'bbox': current_bbox,
                'visible': current_visible,
                'valid': current_valid
            }
            processed_annotations.append(current_anno)
        
        # Build multi-turn tracking conversations
        conversation_data = build_multi_turn_conversations(
            dataset_name=dataset_name,
            exp_str=exp_str,
            frame_paths=frame_paths,
            annotations=processed_annotations,
            image_dimensions=image_dimensions,
            sample_id=i
        )
        
        if conversation_data:
            all_samples.extend(conversation_data)
        else:
            skipped_count += 1
    
    # Create Huggingface Dataset
    print(f"Total conversation samples: {len(all_samples)}")
    print("Validating data structure consistency...")
    
    # 统一处理数据格式以解决PyArrow错误
    for sample in all_samples:
        # 确保image字段总是列表
        if not isinstance(sample["image"], list):
            sample["image"] = [sample["image"]]
        
        # 统一所有消息的content格式
        for message in sample["prompt"]:
            if isinstance(message["content"], str):
                # 将字符串内容转换为列表格式
                message["content"] = [{"text": message["content"]}]
    
    # 创建数据集
    hf_dataset = Dataset.from_list(all_samples)
    
    # Save dataset
    dataset_path = os.path.join(output_dir, "tracking_dataset")
    hf_dataset.save_to_disk(dataset_path)
    
    print("-" * 30)
    print(f"Multi-turn dataset saved to: {dataset_path}")
    print(f"Total samples attempted: {data_len}")
    print(f"Samples skipped: {skipped_count}")
    print(f"Total conversation turns: {len(all_samples)}")
    print("-" * 30)
    
    return hf_dataset


TRACKING_SYSTEM_PROMPT = (
    "You are a professional visual object tracking assistant. Your task is to track specified target objects in a video sequence. "
    "The user will provide an initial frame with the target's bounding box, then you need to find the target's new position in subsequent frames. "
    "Please directly return the target's bounding box coordinates in the format [x1, y1, x2, y2], where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate. "
    "Your answer should be wrapped in <answer>[x1, y1, x2, y2]</answer> tags."
)


def build_multi_turn_conversations(dataset_name, exp_str, frame_paths, annotations, image_dimensions, sample_id):
    """
    Builds a series of multi-turn tracking conversations.
    
    Args:
        dataset_name: Name of the dataset
        exp_str: Object description
        frame_paths: Paths to original frames
        annotations: List of annotation dictionaries
        image_dimensions: List of (width, height) tuples
        sample_id: Sample identifier
        
    Returns:
        List of conversation samples in format compatible with the chat template
    """
    if len(frame_paths) < 2 or len(annotations) < 2:
        return None
    
    # First frame as reference
    ref_frame_path = frame_paths[0]
    ref_anno = annotations[0]
    ref_bbox = ref_anno.get('bbox')
    ref_visible = ref_anno.get('visible', True)
    ref_width, ref_height = image_dimensions[0]
    
    # Create normalized reference bbox
    if not (ref_bbox and ref_visible):
        return None  # Reference object must be visible
    
    ref_norm_bbox = normalize_bbox_xyhw(ref_bbox, ref_width, ref_height)
    
    # System message for all turns - 直接使用字符串格式
    # system_message = {"role": "system", "content": TRACKING_SYSTEM_PROMPT}

    # Create samples for each turn in the conversation
    conversation_samples = []
    
    # First turn (initialization)
    for turn_idx in range(1, len(frame_paths)):
        # Build the conversation history up to this point
        prompt_messages = []
        
        # 图像路径按照对话历史中图像出现的顺序排列
        image_paths_for_turn = []
        
        # 初始消息需要包含图像引用 - 使用content列表格式以兼容chat template
        init_user_content = [
            {"type": "image"},  # 这将被转换为<|vision_start|><|image_pad|><|vision_end|>
            {"text": f"\nThis is the initial frame. Track the object '{exp_str}' with initial bounding box {ref_norm_bbox}."}
        ]
        init_user_msg = {"role": "user", "content": init_user_content}
        prompt_messages.append(init_user_msg)
        image_paths_for_turn.append(frame_paths[0])  # 添加参考帧
        
        # Initial assistant acknowledgement - 直接使用字符串格式
        init_assistant_msg = {"role": "assistant", "content": f"I'll track the {exp_str} starting at position {ref_norm_bbox}."}
        prompt_messages.append(init_assistant_msg)
        
        # Add all intermediate turns if this isn't the first turn
        for prev_idx in range(1, turn_idx):
            prev_width, prev_height = image_dimensions[prev_idx]
            prev_anno = annotations[prev_idx]
            prev_bbox = prev_anno.get('bbox')
            prev_visible = prev_anno.get('visible', False)
            
            # User message for this frame with image - 仍然需要列表格式，因为包含图像
            user_content = [
                {"type": "image"},  # 图像引用
                {"text": f"\nHere's the next frame. Where is the {exp_str} now?"}
            ]
            user_msg = {"role": "user", "content": user_content}
            prompt_messages.append(user_msg)
            image_paths_for_turn.append(frame_paths[prev_idx])  # 添加中间帧
            
            # Assistant response for this frame - 直接使用字符串格式
            if prev_visible and prev_bbox:
                prev_norm_bbox = normalize_bbox_xyhw(prev_bbox, prev_width, prev_height)
                assistant_content = f"The object '{exp_str}' is visible, with a bounding box at {prev_norm_bbox}."
            else:
                assistant_content = "The object is not visible in this frame."
            
            assistant_msg = {"role": "assistant", "content": assistant_content}
            prompt_messages.append(assistant_msg)
        
        # Current turn's user message with image - 仍然需要列表格式，因为包含图像
        current_user_content = [
            {"type": "image"},  # 图像引用
            {"text": f"\nHere's the next frame. Where is the {exp_str} now?"}
        ]
        current_user_msg = {"role": "user", "content": current_user_content}
        prompt_messages.append(current_user_msg)
        image_paths_for_turn.append(frame_paths[turn_idx])  # 添加当前帧
        
        # Current turn's target annotation
        target_anno = annotations[turn_idx]
        target_bbox = target_anno.get('bbox')
        target_visible = target_anno.get('visible', False)
        target_width, target_height = image_dimensions[turn_idx]
        
        # Expected assistant response (solution) - 直接使用字符串格式
        if target_visible and target_bbox:
            target_norm_bbox = normalize_bbox_xyhw(target_bbox, target_width, target_height)
            solution = f"The object '{exp_str}' is visible, with a bounding box at {target_norm_bbox}."
        else:
            solution = "The object is not visible in this frame."
        
        # 创建符合需要的样本格式
        sample = {
            "image": image_paths_for_turn,  # 按对话中图像出现顺序排列的图像路径
            "problem": TRACKING_SYSTEM_PROMPT,  # 系统提示作为问题
            "solution": solution,  # 助手的最终回答作为解决方案
            "prompt": prompt_messages  # 完整的对话历史
        }
        
        conversation_samples.append(sample)
    
    return conversation_samples


if __name__ == "__main__":
    # Set random seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed} for better reproducibility")
    
    # Import moved here to avoid circular imports
    from build_pytorch_dataset import build_dataset
    
    train_dataset, val_dataset = build_dataset()
    
    # For testing, limit the number of samples
    train_dataset.samples_per_epoch = 10
    
    # Build multi-turn dataset
    build_multi_turn_tracking_dataset(train_dataset, output_dir="multi_turn_tracking_dataset")
    # Load and inspect the dataset
    from datasets import load_from_disk
    dataset = load_from_disk("multi_turn_tracking_dataset/tracking_dataset")
    print("Dataset sample:")
    print(dataset[0])
    print(f"Total samples: {len(dataset)}")