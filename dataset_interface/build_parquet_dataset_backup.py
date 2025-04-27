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
        List of conversation samples in GRPO-compatible format
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
    
    # System message for all turns
    system_message = {"role": "system", "content": TRACKING_SYSTEM_PROMPT}

    # Create samples for each turn in the conversation
    conversation_samples = []
    
    # First turn (initialization)
    for turn_idx in range(1, len(frame_paths)):
        # Build the conversation history up to this point
        prompt_messages = [system_message]
        
        # Initial message with reference frame
        init_user_msg = {"role": "user", "content": f"<image>\nThis is the initial frame. Track the object '{exp_str}' with initial bounding box {ref_norm_bbox}."}
        prompt_messages.append(init_user_msg)
        
        # Initial assistant acknowledgement
        init_assistant_msg = {"role": "assistant", "content": f"I'll track the {exp_str} starting at position {ref_norm_bbox}."}
        prompt_messages.append(init_assistant_msg)
        
        # Add all intermediate turns if this isn't the first turn
        for prev_idx in range(1, turn_idx):
            prev_width, prev_height = image_dimensions[prev_idx]
            prev_anno = annotations[prev_idx]
            prev_bbox = prev_anno.get('bbox')
            prev_visible = prev_anno.get('visible', False)
            
            # User message for this frame
            user_msg = {"role": "user", "content": f"<image>\nHere's the next frame. Where is the {exp_str} now?"}
            prompt_messages.append(user_msg)
            
            # Assistant response for this frame
            if prev_visible and prev_bbox:
                prev_norm_bbox = normalize_bbox_xyhw(prev_bbox, prev_width, prev_height)
                assistant_msg = {"role": "assistant", "content": f"The object '{exp_str}' is visible, with a bounding box at {prev_norm_bbox}."}
            else:
                assistant_msg = {"role": "assistant", "content": "The object is not visible in this frame."}
            prompt_messages.append(assistant_msg)
        
        # Current turn's user message
        current_user_msg = {"role": "user", "content": f"<image>\nHere's the next frame. Where is the {exp_str} now?"}
        prompt_messages.append(current_user_msg)
        
        # Current turn's target annotation
        target_anno = annotations[turn_idx]
        target_bbox = target_anno.get('bbox')
        target_visible = target_anno.get('visible', False)
        target_width, target_height = image_dimensions[turn_idx]
        
        # Expected assistant response
        if target_visible and target_bbox:
            target_norm_bbox = normalize_bbox_xyhw(target_bbox, target_width, target_height)
            assistant_response = f"The object '{exp_str}' is visible, with a bounding box at {target_norm_bbox}."
        else:
            assistant_response = "The object is not visible in this frame."
        
        # Create completion
        completion = [
            # current_user_msg,
            {"role": "assistant", "content": assistant_response}
        ]
        
        # 图像路径按照对话历史中<image>出现的顺序排列
        image_paths_for_turn = []
        image_indices = []  # 跟踪每个<image>对应原始frame_paths中的索引
        
        # 添加参考帧
        image_paths_for_turn.append(frame_paths[0])
        image_indices.append(0)
        
        # 添加中间帧
        for prev_idx in range(1, turn_idx):
            image_paths_for_turn.append(frame_paths[prev_idx])
            image_indices.append(prev_idx)
        
        # 添加当前帧
        image_paths_for_turn.append(frame_paths[turn_idx])
        image_indices.append(turn_idx)
        
        # 还可以添加对应关系到对话历史中
        # 例如:
        # init_user_msg = {"role": "user", "content": f"<image index=0>\nThis is..."}
        
        sample = {
            "prompt": prompt_messages,
            "image_paths": image_paths_for_turn,
            # "image_indices": image_indices,
            "completion": completion
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
    print(dataset[1])
    print(dataset[2])
    print(f"Total samples: {len(dataset)}")