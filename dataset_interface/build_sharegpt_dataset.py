import os
from PIL import Image
import json
from tqdm import tqdm
from utils.build_message import build_sft_message
from build_pytorch_dataset import build_dataset
import random
import numpy as np
import torch

from utils.utils import normalize_bbox_xyhw, draw_normed_bbox


def get_bbox_thickness(bbox_width, bbox_height):
    """
    Calculates a dynamic bounding box thickness based on the size of the bounding box.

    Args:
        bbox_width (int): The width of the bounding box.
        bbox_height (int): The height of the bounding box.

    Returns:
        int: The thickness of the bounding box.
    """
    return 1


def build_json_dataset(pytorch_dataset, output_dir="output_datasets", draw_bbox=False):
    """
    Converts a sequential PyTorch Dataset (with dict-of-lists annotations)
    to a multi-turn visual Alpaca format JSON dataset for VQA, storing absolute image paths.

    Args:
        pytorch_dataset (torch.utils.data.Dataset): Your PyTorch Dataset instance
                                                   yielding sequential data.
        output_dir (str): The directory to save the output JSON file.
        draw_bbox (bool): This argument is kept for compatibility but is effectively ignored
                          as images are no longer saved.
    """
    data_len = len(pytorch_dataset)
    all_data = []

    # # Create the output directory for the JSON file if it doesn't exist
    # os.makedirs(output_dir, exist_ok=True)

    skipped_count = 0
    for i in tqdm(range(data_len), desc="Conversion Progress"):

        sample = pytorch_dataset[i]

        frame_paths = sample.get("images")  # Expect list of absolute paths
        annotations_dict = sample.get("anno")
        dataset_name = sample.get("dataset_name")
        exp_str = sample.get("exp_str")
        if exp_str.endswith((" ", "\n", ".")):
            exp_str = exp_str[:-1]

        num_frames = len(frame_paths)

        original_frame_paths_abs = []  # Store original absolute paths
        processed_annotations_list = []
        image_dimensions = []  # Store (width, height) for normalization
        valid_sample = True

        valid_list = annotations_dict.get('valid', [True] * num_frames)

        for frame_idx, frame_path in enumerate(frame_paths):
            try:
                with Image.open(frame_path) as img:
                    img_width, img_height = img.size
                    image_dimensions.append((img_width, img_height))
            except Exception as e:
                print(f"Warning: Could not open image {frame_path} to get dimensions for sample {i}, frame {frame_idx}. Skipping sample. Error: {e}")
                valid_sample = False
                break

            current_bbox = annotations_dict['bbox'][frame_idx].tolist()
            current_visible = annotations_dict['visible'][frame_idx]
            current_valid = valid_list[frame_idx]

            current_anno_dict_for_frame = {
                'bbox': current_bbox,
                'visible': current_visible,
                'valid': current_valid
            }
            processed_annotations_list.append(current_anno_dict_for_frame)
            original_frame_paths_abs.append(frame_path)

        if not valid_sample:
            skipped_count += 1
            continue

        # multi_turn_data = build_multi_turn_message(
        #     dataset_name=dataset_name,
        #     exp_str=exp_str,
        #     frame_paths_abs=original_frame_paths_abs,
        #     annotations=processed_annotations_list,
        #     image_dimensions=image_dimensions
        # )

        one_turn_data = build_oneturn_ontemplate_message(
            dataset_name=dataset_name,
            exp_str=exp_str,
            frame_paths_abs=original_frame_paths_abs,
            annotations=processed_annotations_list,
            image_dimensions=image_dimensions
        )

        # if multi_turn_data:
            # all_data.append(multi_turn_data)
        if one_turn_data:
            all_data.append(one_turn_data)
        else:
            skipped_count += 1

    # json_filename = f"{os.path.basename(output_dir)}_dataset.json"
    json_filename = f"share_gpt_dataset.json"
    output_json_path = os.path.join(output_dir, json_filename)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("-" * 30)
    print(f"Multi-turn Visual Alpaca format dataset saved to: {output_json_path}")
    print(f"Image paths referenced in the JSON are absolute paths to original images.")
    print(f"Total samples attempted: {data_len}")
    print(f"Samples skipped due to errors or validation issues: {skipped_count}")
    print(f"Samples included in JSON: {len(all_data)}")
    print("-" * 30)


def build_multi_turn_message(dataset_name, exp_str, frame_paths_abs, annotations, image_dimensions):
    """
    Builds a multi-turn message structure for sequential object tracking VQA using absolute image paths.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'lasot').
        exp_str (str): The category/object name or description.
        frame_paths_abs (list[str]): List of *absolute* paths to the sequential frame images.
        annotations (list[dict]): List of annotation dicts for each frame. Expected keys:
                                    'bbox' (list[float]), 'visible' (bool), 'valid' (bool, optional).
        image_dimensions (list[tuple]): List of (width, height) tuples for each frame.

    Returns:
        dict: A dictionary containing the multi-turn messages and image paths,
              or None if the input is invalid (e.g., fewer than 2 frames).
    """

    messages = []
    image_paths_for_json = []

    ref_frame_path_abs = frame_paths_abs[0]
    search_frame_path_abs = frame_paths_abs[1]
    ref_frame_basename = os.path.basename(ref_frame_path_abs)
    search_frame_basename = os.path.basename(search_frame_path_abs)

    image_paths_for_json.append(ref_frame_path_abs)
    image_paths_for_json.append(search_frame_path_abs)

    ref_anno = annotations[0]
    ref_bbox = ref_anno.get('bbox')
    ref_visible = ref_anno.get('visible', False)
    ref_width, ref_height = image_dimensions[0]
    normalized_ref_bbox_str = ""
    if ref_bbox is not None and bool(ref_visible):
        norm_bbox = normalize_bbox_xyhw(ref_bbox, ref_width, ref_height)
        normalized_ref_bbox_str = (f" The object in this frame is at normalized coordinates [{norm_bbox[0]}, {norm_bbox[1]}, "
                                    f"{norm_bbox[2]}, {norm_bbox[3]}].")

    # set different instructions for lasot and other datasets
    tracking_instruction = ""
    if dataset_name == 'lasot':
        tracking_instruction = (f"Please locate this object in the following image (<image>{search_frame_basename}). "
                               f"If the object is visible, provide its bounding box. "
                               f"If it's occluded but still present, provide an estimated bounding box. "
                               f"If it has left the frame, state so.")
    else:
        tracking_instruction = (f"Please locate this object in the following image (<image>{search_frame_basename}). "
                               f"Only provide a bounding box if the object is clearly visible. "
                               f"If it's not visible or occluded, state so without providing coordinates.")
    
    user_content_1 = (f"You are an AI assistant for single object tracking. "
                      f"This image (<image>{ref_frame_basename}) shows the object of interest: '{exp_str}'."
                      f"{normalized_ref_bbox_str} {tracking_instruction}")
    messages.append({"role": "user", "content": user_content_1})

    target_anno = annotations[1]
    target_bbox = target_anno.get('bbox')
    target_visible = target_anno.get('visible', False)
    target_valid = target_anno.get('valid', True)
    target_width, target_height = image_dimensions[1]
    normalized_target_bbox = None

    if target_bbox:
        normalized_target_bbox = normalize_bbox_xyhw(target_bbox, target_width, target_height)

    assistant_content_1 = ""
    if dataset_name == 'lasot':
        if target_visible and normalized_target_bbox:
            assistant_content_1 = (f"The object '{exp_str}' is visible, with a bounding box at "
                                    f"[{normalized_target_bbox[0]}, {normalized_target_bbox[1]}, "
                                    f"{normalized_target_bbox[2]}, {normalized_target_bbox[3]}].")
        elif target_valid and normalized_target_bbox:
            assistant_content_1 = (f"The object '{exp_str}' is present but occluded. Its estimated bounding box is "
                                    f"[{normalized_target_bbox[0]}, {normalized_target_bbox[1]}, "
                                    f"{normalized_target_bbox[2]}, {normalized_target_bbox[3]}].")
        else:
            assistant_content_1 = f"The object '{exp_str}' has left the frame or its location could not be determined."
    else:
        if target_visible and normalized_target_bbox:
            assistant_content_1 = (f"The object '{exp_str}' is visible, with a bounding box at "
                                    f"[{normalized_target_bbox[0]}, {normalized_target_bbox[1]}, "
                                    f"{normalized_target_bbox[2]}, {normalized_target_bbox[3]}].")
        else:
            assistant_content_1 = f"The object '{exp_str}' is not visible."

    messages.append({"role": "assistant", "content": assistant_content_1})

    for i in range(1, len(frame_paths_abs) - 1):
        current_frame_idx = i
        next_frame_idx = i + 1

        next_frame_path_abs = frame_paths_abs[next_frame_idx]
        next_frame_basename = os.path.basename(next_frame_path_abs)

        image_paths_for_json.append(next_frame_path_abs)

        user_content_next = f"Now, find the object '{exp_str}' in the following image (<image>{next_frame_basename})."
        messages.append({"role": "user", "content": user_content_next})

        target_anno = annotations[next_frame_idx]
        target_bbox = target_anno.get('bbox')
        target_visible = target_anno.get('visible', False)
        target_valid = target_anno.get('valid', True)
        target_width, target_height = image_dimensions[next_frame_idx]
        normalized_target_bbox = None
        if target_bbox:
            normalized_target_bbox = normalize_bbox_xyhw(target_bbox, target_width, target_height)

        assistant_content_next = ""
        if dataset_name == 'lasot':
            if target_visible and normalized_target_bbox:
                assistant_content_next = (f"The object '{exp_str}' is visible, with a bounding box at "
                                          f"[{normalized_target_bbox[0]}, {normalized_target_bbox[1]}, "
                                          f"{normalized_target_bbox[2]}, {normalized_target_bbox[3]}].")
            elif target_valid and normalized_target_bbox:
                assistant_content_next = (f"The object '{exp_str}' is present but occluded. Its estimated bounding box is "
                                          f"[{normalized_target_bbox[0]}, {normalized_target_bbox[1]}, "
                                          f"{normalized_target_bbox[2]}, {normalized_target_bbox[3]}].")
            else:
                assistant_content_next = f"The object '{exp_str}' has left the frame or its location could not be determined."
        else:
            if target_visible and normalized_target_bbox:
                assistant_content_next = (f"The object '{exp_str}' is visible, with a bounding box at "
                                          f"[{normalized_target_bbox[0]}, {normalized_target_bbox[1]}, "
                                          f"{normalized_target_bbox[2]}, {normalized_target_bbox[3]}].")
            else:
                assistant_content_next = f"The object '{exp_str}' is not visible."

        messages.append({"role": "assistant", "content": assistant_content_next})

    message_structure = {
        "messages": messages,
        "images": image_paths_for_json
    }

    return message_structure




def build_oneturn_ontemplate_message(dataset_name, exp_str, frame_paths_abs, annotations, image_dimensions):
    """
    构建单轮多帧对象跟踪消息结构，使用第一帧作为模板帧，模型一次性预测所有帧上的目标位置。
    
    Args:
        dataset_name (str): 数据集名称(例如, 'lasot')
        exp_str (str): 类别/对象名称或描述
        frame_paths_abs (list[str]): 序列帧图像的绝对路径列表
        annotations (list[dict]): 每一帧的标注字典列表。期望的键:
                                 'bbox' (list[float]), 'visible' (bool), 'valid' (bool, 可选)
        image_dimensions (list[tuple]): 每一帧的(宽度, 高度)元组列表
        
    Returns:
        dict: 包含单轮对话消息和图像路径的字典，如果输入无效则返回None
    """
    
    
    
    messages = []
    image_paths_for_json = []
    

    for frame_path in frame_paths_abs:
        image_paths_for_json.append(frame_path)
    

    ref_anno = annotations[0]
    ref_bbox = ref_anno.get('bbox')
    ref_visible = ref_anno.get('visible', False)
    ref_width, ref_height = image_dimensions[0]
    
    # 格式化参考帧的边界框字符串
    normalized_ref_bbox_str = ""
    if ref_bbox is not None and bool(ref_visible):
        norm_bbox = normalize_bbox_xyhw(ref_bbox, ref_width, ref_height)
        normalized_ref_bbox_str = (f" The object in this frame is at normalized coordinates [{norm_bbox[0]}, {norm_bbox[1]}, "
                                   f"{norm_bbox[2]}, {norm_bbox[3]}].")
    
    

    system_content = (f"You are an AI assistant for single object tracking. ")
    user_content = (f"This image (<image>) shows the object of interest: '{exp_str}'."
                    f"{normalized_ref_bbox_str} ")

    
    user_content += f"Please track this object across all frames:"
    

    for i in range(1, len(frame_paths_abs)):
        user_content += f" (<image>)"

    
    user_content += (f". For each frame, reply in exactly this format:\n"
                    f"Frame 1: [x1, y1, x2, y2] or \"not visible\"\n"
                    f"Frame 2: [x1, y1, x2, y2] or \"not visible\"\n"
                    f"... and so on for all frames.")
    messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    
    assistant_content = ""
    
    for i in range(len(frame_paths_abs)):
        target_anno = annotations[i]
        target_bbox = target_anno.get('bbox')
        target_visible = target_anno.get('visible', False)
        target_valid = target_anno.get('valid', True)
        target_width, target_height = image_dimensions[i]
        
        normalized_target_bbox = None
        if target_bbox:
            normalized_target_bbox = normalize_bbox_xyhw(target_bbox, target_width, target_height)
        
        assistant_content += f"Frame {i+1}: "
        
        if dataset_name == 'lasot':
            if target_visible and normalized_target_bbox:
                assistant_content += f"[{normalized_target_bbox[0]}, {normalized_target_bbox[1]}, {normalized_target_bbox[2]}, {normalized_target_bbox[3]}]"
            elif target_valid and normalized_target_bbox:
                assistant_content += f"[{normalized_target_bbox[0]}, {normalized_target_bbox[1]}, {normalized_target_bbox[2]}, {normalized_target_bbox[3]}]"
            else:
                assistant_content += "not visible"
        else:
            if target_visible and normalized_target_bbox:
                assistant_content += f"[{normalized_target_bbox[0]}, {normalized_target_bbox[1]}, {normalized_target_bbox[2]}, {normalized_target_bbox[3]}]"
            else:
                assistant_content += "not visible"
                
        if i < len(frame_paths_abs) - 1:
            assistant_content += "\n"
    
    messages.append({"role": "assistant", "content": assistant_content})
    
    message_structure = {
        "messages": messages,
        "images": image_paths_for_json
    }
    
    return message_structure

if __name__ == "__main__":
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
    train_dataset, val_dataset = build_dataset()
    train_dataset.samples_per_epoch = 10

    output_json_dir = "/data1/lihaobo/tracking/output_dir"
    build_json_dataset(train_dataset, output_dir=output_json_dir, draw_bbox=False)



