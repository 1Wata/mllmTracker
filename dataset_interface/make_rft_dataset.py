import os
import argparse
from PIL import Image
import json
from tqdm import tqdm
import random
import numpy as np
import torch
import shutil
from datasets import Dataset
from utils.utils import normalize_bbox_xyhw
from make_crop_dataset import crop_and_pad_template, crop_and_pad_search, jitter_bbox, convert_bbox_format, is_bbox_fully_visible


def normalize_and_scale_bbox_abs(bbox_abs, img_w, img_h):
    """
    Normalizes an absolute bbox [x1, y1, x2, y2] and scales to 1000, then converts to int.
    Args:
        bbox_abs: List or tuple [x1, y1, x2, y2] in absolute pixel coordinates.
        img_w: Width of the image.
        img_h: Height of the image.
    Returns:
        List of ints [nx1, ny1, nx2, ny2] scaled to 1000.
    """
    if img_w == 0 or img_h == 0: # Avoid division by zero
        print(f"Warning: Image width or height is zero (w={img_w}, h={img_h}). Returning zero bbox.")
        return [0, 0, 0, 0]
    return [
        int(bbox_abs[0] / img_w * 1000),
        int(bbox_abs[1] / img_h * 1000),
        int(bbox_abs[2] / img_w * 1000),
        int(bbox_abs[3] / img_h * 1000)
    ]


def build_one_turn_tracking_dataset(pytorch_dataset, output_dir="one_turn_tracking_dataset_cropped",
                                    template_frames=1, scale=2.0, search_scale=4.0, resize=320, no_crop=False, normalize_bbox=False):
    """
    构建一个跟踪数据集，可选是否使用裁剪图像。
    
    Args:
        pytorch_dataset: PyTorch数据集实例
        output_dir: 输出目录路径
        template_frames: 使用的模板帧数量
        scale: 模板帧裁剪的缩放因子
        search_scale: 搜索帧裁剪的缩放因子
        resize: 裁剪后图像的尺寸
        no_crop: 是否不裁剪图像，直接使用原始图像路径
        normalize_bbox: 是否对边界框进行归一化 (乘以1000并取整)
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 只有在需要裁剪时才创建裁剪图像目录
    cropped_images_base_dir = None
    if not no_crop:
        cropped_images_base_dir = os.path.join(output_dir, "cropped_images")
        os.makedirs(cropped_images_base_dir, exist_ok=True)
    
    # 用于存储数据集样本的列表
    all_samples = []
    skipped_count = 0
    processed_count = 0
    
    # 根据 normalize_bbox 参数确定实际使用的系统提示
    current_tracking_system_prompt = get_tracking_system_prompt(use_thinking=False, normalize_bbox_for_system_prompt=normalize_bbox)
    
    # 确定要处理的样本数量
    if hasattr(pytorch_dataset, 'samples_per_epoch') and pytorch_dataset.samples_per_epoch is not None:
        data_len = min(len(pytorch_dataset), pytorch_dataset.samples_per_epoch)
        print(f"Processing {data_len} samples based on samples_per_epoch setting.")
    else:
        data_len = len(pytorch_dataset)
        print(f"Processing all {data_len} samples in the dataset.")
    
    # 辅助函数：安全删除目录
    def safe_remove_dir(dir_path):
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"Removed directory: {dir_path}")
            except OSError as e:
                print(f"Error removing directory {dir_path}: {e}")
    
    for i in tqdm(range(data_len), desc="Building One-turn Dataset" + (" with Original Images" if no_crop else " with Cropped Images")):
        # 只有在需要裁剪时才创建样本目录
        sample_dir = None
        if not no_crop:
            sample_dir = os.path.join(cropped_images_base_dir, f"sample_{i:06d}")
            os.makedirs(sample_dir, exist_ok=True)
        
        sample = pytorch_dataset[i]
        
        template_frame_paths = sample.get("template_images", [])
        template_anno_dict = sample.get("template_anno", {})
        search_frame_paths = sample.get("search_images", [])
        search_anno_dict = sample.get("search_anno", {})
        dataset_name = sample.get("dataset_name", "unknown")
        exp_str = sample.get("exp_str", "object")
        
        if exp_str.endswith((" ", "\n", ".")):
            exp_str = exp_str[:-1]
        
        # 检查序列是否有足够的帧
        if len(template_frame_paths) == 0 or len(search_frame_paths) < 1: # Need at least one template initially
            if not no_crop and sample_dir:
                safe_remove_dir(sample_dir)
            skipped_count += 1
            continue
            
        # 获取有效帧列表
        template_valid_list = template_anno_dict.get('valid', [True] * len(template_frame_paths))
        search_valid_list = search_anno_dict.get('valid', [True] * len(search_frame_paths))

        # --- 组合模板帧和搜索帧 ---
        frame_paths = template_frame_paths + search_frame_paths
        num_frames = len(frame_paths)

        # 组合标注信息
        annotations_dict = {
            'bbox': template_anno_dict.get('bbox', []) + search_anno_dict.get('bbox', []),
            'visible': template_valid_list + search_valid_list
        }

        # --- 获取所有帧的原始尺寸 (如果需要归一化或裁剪) ---
        all_frame_dims = []
        if normalize_bbox or not no_crop:
            for frame_path_for_dim in frame_paths:
                try:
                    with Image.open(frame_path_for_dim) as img:
                        img_w, img_h = img.size
                        all_frame_dims.append((img_w, img_h))
                except FileNotFoundError:
                    print(f"Warning: Frame path not found: {frame_path_for_dim} for sample {i}. Skipping dimension retrieval for this frame.")
                    all_frame_dims.append((0,0)) # Add placeholder to maintain index correspondence
                    # Consider skipping sample if critical dimensions are missing
                    # For now, normalization function will handle 0 dimensions

        # --- Determine template indices based on probability ---
        # 10% chance to use only the last template frame if more than one is requested and available
        use_last_template_only = random.random() < 0.1 and template_frames > 1 and len(template_frame_paths) > 0

        if no_crop:
            # --- Determine template indices ---
            if use_last_template_only:
                template_indices = [len(template_frame_paths) - 1]
            else:
                # Use up to template_frames, but no more than available
                num_templates_to_use = min(template_frames, len(template_frame_paths))
                template_indices = list(range(num_templates_to_use))

            search_indices = list(range(len(template_frame_paths), num_frames))

            # 使用原始模板帧路径
            cropped_template_paths = [template_frame_paths[idx] for idx in template_indices]

            # 使用原始搜索帧路径
            cropped_search_paths = [frame_paths[idx] for idx in search_indices]

            # 使用原始边界框
            cropped_search_bboxes = []
            for s_idx_loop, frame_path_global_idx in enumerate(search_indices):
                bbox_raw = annotations_dict['bbox'][frame_path_global_idx]
                bbox_raw = bbox_raw.tolist() if isinstance(bbox_raw, torch.Tensor) else bbox_raw
                search_bbox_abs = convert_bbox_format(bbox_raw)  # 转换为 [x1, y1, x2, y2] 格式
                
                if normalize_bbox:
                    if frame_path_global_idx < len(all_frame_dims):
                        img_w, img_h = all_frame_dims[frame_path_global_idx]
                        processed_bbox = normalize_and_scale_bbox_abs(search_bbox_abs, img_w, img_h)
                    else:
                        print(f"Warning: Missing dimensions for frame_path_global_idx {frame_path_global_idx} in no_crop mode for sample {i}. Using [0,0,0,0] for normalized bbox.")
                        processed_bbox = [0,0,0,0]
                else:
                    processed_bbox = [int(c) for c in search_bbox_abs]
                cropped_search_bboxes.append(processed_bbox)

        else:
            # 裁剪模式下的处理逻辑
            # --- 获取所有帧的原始数据 ---
            all_bboxes_orig_raw = []
            # all_frame_dims is already populated if not no_crop

            for frame_idx, frame_path in enumerate(frame_paths):
                # Dimensions already read into all_frame_dims
                bbox_raw = annotations_dict['bbox'][frame_idx].tolist() if isinstance(annotations_dict['bbox'][frame_idx], torch.Tensor) else annotations_dict['bbox'][frame_idx]
                convert_bbox_format(bbox_raw)  # 检查边界框格式是否有效
                all_bboxes_orig_raw.append(bbox_raw)

            # --- Determine template indices ---
            if use_last_template_only:
                template_indices = [len(template_frame_paths) - 1]
            else:
                # Use up to template_frames, but no more than available
                num_templates_to_use = min(template_frames, len(template_frame_paths))
                template_indices = list(range(num_templates_to_use))

            # --- 处理选定的模板帧 ---
            if not template_indices: # Should not happen if initial check passed, but safety first
                 print(f"Skipping sample {i} because no template indices were selected.")
                 if sample_dir: safe_remove_dir(sample_dir)
                 skipped_count += 1
                 continue

            template_bboxes_orig_selected = [convert_bbox_format(all_bboxes_orig_raw[idx]) for idx in template_indices]
            template_bboxes_jittered_selected = [jitter_bbox(bbox, jitter_scale=0.05) for bbox in template_bboxes_orig_selected]

            cropped_template_paths = []
            for t_idx, template_idx in enumerate(template_indices): # Iterate using selected indices
                cropped_template = crop_and_pad_template(
                    frame_paths[template_idx],
                    template_bboxes_jittered_selected[t_idx], # Use corresponding jittered bbox
                    scale=scale,
                    resize=resize
                )
                template_filename = f"template_{template_idx:03d}.jpg" # Use original index for filename
                template_save_path = os.path.join(sample_dir, template_filename)
                cropped_template.save(template_save_path)
                cropped_template_paths.append(template_save_path)

            # --- 处理所有搜索帧 (除了模板帧) ---
            # Search indices start after all *original* template frames
            search_indices = list(range(len(template_frame_paths), num_frames))
            cropped_search_paths = []
            cropped_search_bboxes = []

            # Use the first *selected* jittered bbox as reference
            reference_bbox_for_search_crop = template_bboxes_jittered_selected[0]

            for search_idx in search_indices:
                search_bbox_orig = convert_bbox_format(all_bboxes_orig_raw[search_idx])

                try:
                    cropped_search_img, search_bbox_new_abs, crop_region, _ = crop_and_pad_search(
                        frame_paths[search_idx],
                        reference_bbox_for_search_crop,
                        search_bbox_orig,
                        scale=search_scale,
                        resize=resize
                    )

                    # Calculate relative index for filename (index within search frames)
                    search_filename_idx = search_idx - len(template_frame_paths)
                    search_filename = f"search_{search_filename_idx:03d}.jpg"
                    search_save_path = os.path.join(sample_dir, search_filename)
                    cropped_search_img.save(search_save_path)

                    cropped_search_paths.append(search_save_path)
                    
                    if normalize_bbox:
                        # For cropped images, normalization is relative to the cropped image size (resize x resize)
                        processed_bbox = normalize_and_scale_bbox_abs(search_bbox_new_abs, resize, resize)
                    else:
                        processed_bbox = [int(c) for c in search_bbox_new_abs]
                    cropped_search_bboxes.append(processed_bbox)
                except Exception as e:
                    print(f"Warning: Error processing search frame {search_idx} for sample {i}: {e}")
                    continue # Skip this search frame

        # --- 检查是否没有成功处理任何搜索帧 ---
        if not cropped_search_paths:
            print(f"Skipping sample {i} because no search frames were successfully processed.")
            if not no_crop and sample_dir:
                safe_remove_dir(sample_dir)
            skipped_count += 1
            continue

        image_paths_for_sample = cropped_template_paths + cropped_search_paths

        # 构建用户提示
        init_user_content = []

        # 添加模板帧
        for _ in range(len(cropped_template_paths)):
            init_user_content.append({"type": "image"})


        for idx, template_original_idx in enumerate(template_indices):
            if no_crop:
                # 原始坐标系下的bbox
                bbox_raw_list = template_anno_dict.get('bbox', [])
                if template_original_idx < len(bbox_raw_list):
                    bbox_raw = bbox_raw_list[template_original_idx]
                    if isinstance(bbox_raw, torch.Tensor):
                        bbox_raw = bbox_raw.tolist()
                    template_bbox_abs = convert_bbox_format(bbox_raw)  # 转换为 [x1, y1, x2, y2] 格式
                    
                    if template_bbox_abs:
                        if normalize_bbox:
                            if template_original_idx < len(all_frame_dims):
                                img_w, img_h = all_frame_dims[template_original_idx]
                                norm_bbox = normalize_and_scale_bbox_abs(template_bbox_abs, img_w, img_h)
                                bbox_text = f"\nThe normalized bounding box (scaled to 1000) for template frame {idx+1} is: [{norm_bbox[0]}, {norm_bbox[1]}, {norm_bbox[2]}, {norm_bbox[3]}]."
                            else:
                                bbox_text = f"\nWarning: Missing dimensions for template_original_idx {template_original_idx} in no_crop mode for sample {i}. Cannot provide normalized bbox."
                        else:
                            bbox_text = f"\nThe bounding box for template frame {idx+1} is: [{int(template_bbox_abs[0])}, {int(template_bbox_abs[1])}, {int(template_bbox_abs[2])}, {int(template_bbox_abs[3])}]."
                        init_user_content.append({"text": bbox_text})
                else:
                    print(f"Warning: template_original_idx {template_original_idx} out of bounds for template_anno_dict['bbox'] (len {len(bbox_raw_list)}) for sample {i}.")

            else:
                # 裁剪模式下不需要添加，因为已经通过裁剪图像隐式传达了目标位置
                pass
        # 描述物体
        template_text = f"\nThese are the template frames showing the object '{exp_str}'."
        init_user_content.append({"text": template_text})

        # 添加所有搜索帧
        for _ in range(len(cropped_search_paths)):
            init_user_content.append({"type": "image"})

        # 添加跟踪指令，根据搜索帧数量调整语言
        tracking_instruction = f" Please track the object '{exp_str}' in "
        if len(cropped_search_paths) > 1:
            tracking_instruction += f"these {len(cropped_search_paths)} search frames. "
            tracking_instruction += f"For each of the {len(cropped_search_paths)} frames, provide a separate bounding box. "
        else:
            tracking_instruction += "the next frame. "
            tracking_instruction += "Provide a bounding box for the last frame. "
        
        if normalize_bbox:
            tracking_instruction += "The bounding box should be in normalized [x1, y1, x2, y2] format, where coordinates are integers scaled to the range [0, 1000]. "
        else:
            tracking_instruction += "The bounding box should be in [x1, y1, x2, y2] format with absolute pixel coordinates. "
        # tracking_instruction += "Wrap each answer in <answer></answer> tags."
        init_user_content.append({"text": tracking_instruction})

        init_user_msg = {"role": "user", "content": init_user_content}
        prompt_messages = [init_user_msg]

        # 构建解决方案（仅包含搜索帧的跟踪结果）
        solution = f"Tracking results for '{exp_str}':\n\n"
        any_search_invisible = False

        # 只包含搜索帧结果，使用相应的坐标
        temp_solution_lines = []
        for idx, search_bbox in enumerate(cropped_search_bboxes):
            search_idx_relative = idx  # 在no_crop模式下直接使用idx
            if not no_crop:
                search_idx_relative = search_indices[idx] - len(template_frame_paths)
            
            # 检查原始注释中的可见性标志
            if search_idx_relative < len(search_valid_list):
                search_visible = search_valid_list[search_idx_relative]
            else:
                print(f"Warning: search_idx {search_idx_relative} out of bounds for visibility list (len {len(search_valid_list)}) for sample {i}. Treating as visible.")
                search_visible = True
            
            # 为多帧情况添加帧标识
            if len(cropped_search_paths) > 1:
                frame_prefix = f"Frame {idx+1}: "
            else:
                frame_prefix = ""
                
            if search_visible:
                temp_solution_lines.append(f"{frame_prefix}<answer>[{int(search_bbox[0])}, {int(search_bbox[1])}, {int(search_bbox[2])}, {int(search_bbox[3])}]</answer>\n")
            else:
                temp_solution_lines.append(f"{frame_prefix}<answer>[0, 0, 0, 0]</answer>\n")
                any_search_invisible = True

        # --- 检查是否有不可见的搜索帧 ---
        if any_search_invisible:
            print(f"Skipping sample {i} because it contains invisible search frames.")
            if not no_crop:
                safe_remove_dir(sample_dir)
            skipped_count += 1
            continue

        # --- 如果所有搜索帧都可见，则完成solution字符串 ---
        solution += "".join(temp_solution_lines)

        # 创建样本
        sample_data = {
            "image": image_paths_for_sample,
            "problem": current_tracking_system_prompt,
            "solution": solution,
            "prompt": prompt_messages
        }

        all_samples.append(sample_data)
        processed_count += 1
    
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
    
    # 保存数据集
    dataset_path = os.path.join(output_dir, "tracking_dataset")
    hf_dataset.save_to_disk(dataset_path)
    
    print("-" * 30)
    print(f"Dataset saved to: {dataset_path}")
    print(f"Mode: {'Original images (no crop)' if no_crop else 'Cropped images'}")
    print(f"Total samples attempted: {data_len}")
    print(f"Samples skipped: {skipped_count}")
    print(f"Total conversation samples: {len(all_samples)}")
    print("-" * 30)
    
    return hf_dataset

def get_tracking_system_prompt(use_thinking=False, normalize_bbox_for_system_prompt=False):
    """获取追踪任务的系统提示"""
    base_prompt = (
        "You are a professional visual object tracking assistant. Your task is to track a specified target object. "
        "The user will provide template frames showing the target object. "
        "First, identify the target in the template frames. Then, locate the target's position in the following search frames."
    )
    
    bbox_format_instruction = "the target's bounding box coordinates in the format [x1, y1, x2, y2], where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate."
    if normalize_bbox_for_system_prompt:
        bbox_format_instruction = "the target's bounding box coordinates in normalized [x1, y1, x2, y2] format, where coordinates are integers scaled to the range [0, 1000]."
    
    if use_thinking:
        return base_prompt + (
            " You should first think about how to locate the target by analyzing visual features, then provide your answer. "
            "Put your thinking process inside <thinking>...</thinking> tags. "
            f"Your final answer should be {bbox_format_instruction} "
            "Wrap your final answer in <answer>[x1, y1, x2, y2]</answer> tags."
        )
    else:
        return base_prompt + (
            f" Please directly return {bbox_format_instruction} "
            "Your answer should be wrapped in <answer>[x1, y1, x2, y2]</answer> tags."
        )


TRACKING_SYSTEM_PROMPT = get_tracking_system_prompt(False)

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="构建使用裁剪图像的一轮跟踪数据集")
    
    # 添加命令行参数
    parser.add_argument("--samples_per_epoch", type=int, default=20)
    parser.add_argument("--output_dir", default='/data1/lihaobo/tracking/test', type=str)
    
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--search_scale", type=float, default=3.0)
    parser.add_argument("--resize", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--template_frames", type=int, default=2)
    parser.add_argument("--search_frames", type=int, default=1)
    parser.add_argument("--no_crop", default=True, help="是否不裁剪图像，直接使用原始图像路径")
    parser.add_argument("--normalize_bbox", default=True, help="是否使用归一化的边界框。如果是，坐标将被归一化并乘以1000取整。")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置随机种子
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed} for better reproducibility")
    
    # 导入构建数据集的函数
    from build_pytorch_dataset import build_dataset
    
    # 构建PyTorch数据集
    train_dataset = build_dataset(num_search_frames=args.search_frames, num_template_frames=args.template_frames)


    train_dataset.samples_per_epoch = args.samples_per_epoch
    print(f"Processing {args.samples_per_epoch} samples per epoch")

    
    # 构建使用裁剪图像的一轮跟踪数据集
    output_dir = args.output_dir
    print(f"Output directory set to: {output_dir}")
    
    build_one_turn_tracking_dataset(
        train_dataset,
        output_dir=output_dir,
        template_frames=args.template_frames,
        scale=args.scale,
        search_scale=args.search_scale,
        resize=args.resize,
        no_crop=args.no_crop,
        normalize_bbox=args.normalize_bbox
    )
    
    # 加载并检查数据集
    from datasets import load_from_disk
    dataset = load_from_disk(f"{output_dir}/tracking_dataset")
    print("Dataset sample:")
    print(dataset[2])
    print(f"Total samples: {len(dataset)}")