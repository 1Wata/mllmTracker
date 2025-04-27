import os
from PIL import Image
import json
from tqdm import tqdm
import random
import numpy as np
import torch
from datasets import Dataset
from utils.utils import normalize_bbox_xyhw

def build_one_turn_tracking_dataset(pytorch_dataset, output_dir="one_turn_tracking_dataset"):
    """
    Converts a sequential PyTorch Dataset to a Huggingface Dataset in one-turn conversation format
    for single object tracking, where one template frame is used to track objects in multiple frames.
    
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
    for i in tqdm(range(data_len), desc="Building One-turn Dataset"):
        sample = pytorch_dataset[i]
        
        frame_paths = sample.get("images")
        annotations_dict = sample.get("anno")
        dataset_name = sample.get("dataset_name", "unknown")
        exp_str = sample.get("exp_str")
        if exp_str.endswith((" ", "\n", ".")):
            exp_str = exp_str[:-1]
    
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
        
        # Build one-turn tracking conversation
        conversation_data = build_one_turn_conversation(
            dataset_name=dataset_name,
            exp_str=exp_str,
            frame_paths=frame_paths,
            annotations=processed_annotations,
            image_dimensions=image_dimensions,
            sample_id=i
        )
        
        if conversation_data:
            all_samples.append(conversation_data)
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
    print(f"One-turn dataset saved to: {dataset_path}")
    print(f"Total samples attempted: {data_len}")
    print(f"Samples skipped: {skipped_count}")
    print(f"Total conversation samples: {len(all_samples)}")
    print("-" * 30)
    
    return hf_dataset


TRACKING_SYSTEM_PROMPT = (
    "You are a professional visual object tracking assistant. Your task is to track specified target objects in a video sequence. "
    "The user will provide an initial frame with the target's bounding box, then you need to find the target's new position in subsequent frames. "
    "Please provide the target's bounding box coordinates for each frame in the format [x, y, width, height]."
    "If the object is not visible in any frame, explicitly state so for that frame."
)


def build_one_turn_conversation(dataset_name, exp_str, frame_paths, annotations, image_dimensions, sample_id):
    """
    Builds a one-turn conversation where one template frame is used to predict multiple frames.
    
    Args:
        dataset_name: Name of the dataset
        exp_str: Object description
        frame_paths: Paths to original frames
        annotations: List of annotation dictionaries
        image_dimensions: List of (width, height) tuples
        sample_id: Sample identifier
        
    Returns:
        A conversation sample in format compatible with the chat template
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
    
    # Build prompt messages for one-turn conversation
    prompt_messages = []
    
    # 收集所有要在这个对话中使用的图像路径
    image_paths_for_sample = []
    
    # 添加模板帧
    image_paths_for_sample.append(frame_paths[0])
    
    # 初始用户消息 - 显示模板帧和目标对象
    init_user_content = [
        {"type": "image"},  # 这将被转换为<|vision_start|><|image_pad|><|vision_end|>
        {"text": f"\nThis is the initial frame showing the object '{exp_str}' with bounding box {ref_norm_bbox}."}
    ]
    
    # 添加所有需要跟踪的帧（除了第一帧）
    for i in range(1, len(frame_paths)):
        image_paths_for_sample.append(frame_paths[i])
        init_user_content.append({"type": "image"})
        
    # 添加跟踪指令 - 修改指令以使用[0,0,0,0]表示不可见
    init_user_content.append({"text": f" Please track the object '{exp_str}' across all these frames. "
                                     f"For each frame, provide the normalized bounding box [x, y, width, height]. "
                                     f"Use [0, 0, 0, 0] if the object is not visible."})
    
    # 添加用户消息到对话
    init_user_msg = {"role": "user", "content": init_user_content}
    prompt_messages.append(init_user_msg)
    
    # 构建期望的助手回复（包含所有帧的位置信息）
    solution = f"Tracking results for '{exp_str}':\n\n"
    
    # 为每一帧添加跟踪结果 - 使用[0,0,0,0]替代"not visible"
    for i in range(len(frame_paths)):
        frame_anno = annotations[i]
        frame_bbox = frame_anno.get('bbox')
        frame_visible = frame_anno.get('visible', False)
        frame_width, frame_height = image_dimensions[i]
        
        solution += f"Frame {i+1}: "
        
        if frame_visible and frame_bbox:
            frame_norm_bbox = normalize_bbox_xyhw(frame_bbox, frame_width, frame_height)
            solution += f"[{frame_norm_bbox[0]}, {frame_norm_bbox[1]}, {frame_norm_bbox[2]}, {frame_norm_bbox[3]}]\n"
        else:
            # 用[0,0,0,0]替代"not visible"
            solution += "[0, 0, 0, 0]\n"
    
    # 创建符合需要的样本格式
    sample = {
        "image": image_paths_for_sample,  # 所有图像的路径
        "problem": TRACKING_SYSTEM_PROMPT,  # 系统提示作为问题
        "solution": solution,  # 助手的回答作为解决方案
        "prompt": prompt_messages  # 完整的对话历史
    }
    
    return sample


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
    
    # Build one-turn dataset
    build_one_turn_tracking_dataset(train_dataset, output_dir="one_turn_tracking_dataset")
    
    # Load and inspect the dataset
    from datasets import load_from_disk
    dataset = load_from_disk("one_turn_tracking_dataset/tracking_dataset")
    print("Dataset sample:")
    print(dataset[0])
    print(f"Total samples: {len(dataset)}")