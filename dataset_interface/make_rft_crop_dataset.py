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
from make_crop_dataset.utils import crop_and_pad_template, crop_and_pad_search, jitter_bbox, convert_bbox_format

def is_bbox_fully_visible(bbox, img_width, img_height):
    """检查边界框 [x1, y1, x2, y2] 是否完全在图像范围内。"""
    if not (bbox and len(bbox) == 4):
        return False
    x1, y1, x2, y2 = bbox
    if x1 >= x2 or y1 >= y2:
        return False
    return x1 >= 0 and y1 >= 0 and x2 <= img_width and y2 <= img_height

def build_one_turn_tracking_dataset_cropped(pytorch_dataset, output_dir="one_turn_tracking_dataset_cropped",
                                          template_frames=1, scale=2.0, search_scale=4.0, resize=320):
    """
    构建一个使用裁剪图像的一轮跟踪数据集。
    与build_rft_dataset-oneturn相同的格式，但使用裁剪的图像。
    
    Args:
        pytorch_dataset: PyTorch数据集实例
        output_dir: 输出目录路径
        template_frames: 使用的模板帧数量
        scale: 模板帧裁剪的缩放因子
        search_scale: 搜索帧裁剪的缩放因子
        resize: 裁剪后图像的尺寸
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    cropped_images_base_dir = os.path.join(output_dir, "cropped_images")
    os.makedirs(cropped_images_base_dir, exist_ok=True)
    
    # 用于存储数据集样本的列表
    all_samples = []
    
    skipped_count = 0
    processed_count = 0
    
    # 确定要处理的样本数量
    if hasattr(pytorch_dataset, 'samples_per_epoch') and pytorch_dataset.samples_per_epoch is not None:
        data_len = min(len(pytorch_dataset), pytorch_dataset.samples_per_epoch)
        print(f"Processing {data_len} samples based on samples_per_epoch setting.")
    else:
        data_len = len(pytorch_dataset)
        print(f"Processing all {data_len} samples in the dataset.")
    
    # 处理每个样本
    for i in tqdm(range(data_len), desc="Building One-turn Dataset with Cropped Images"):
        try:
            sample = pytorch_dataset[i]
            
            frame_paths = sample.get("images")
            annotations_dict = sample.get("anno")
            dataset_name = sample.get("dataset_name", "unknown")
            exp_str = sample.get("exp_str", f"sequence_{i}")
            if exp_str.endswith((" ", "\n", ".")):
                exp_str = exp_str[:-1]
            
            # 检查序列是否有足够的帧
            num_frames = len(frame_paths)
            if num_frames < template_frames + 1:
                skipped_count += 1
                continue
            
            # 获取有效帧列表
            valid_list = annotations_dict.get('valid', [True] * num_frames)
            
            # --- 获取所有帧的原始数据 ---
            all_bboxes_orig_raw = []
            all_frame_dims = []
            skip_sample = False
            
            for frame_idx, frame_path in enumerate(frame_paths):
                try:
                    with Image.open(frame_path) as img:
                        img_w, img_h = img.size
                        all_frame_dims.append((img_w, img_h))
                    
                    bbox_raw = annotations_dict['bbox'][frame_idx].tolist() if isinstance(annotations_dict['bbox'][frame_idx], torch.Tensor) else annotations_dict['bbox'][frame_idx]
                    bbox = convert_bbox_format(bbox_raw)  # [x1, y1, x2, y2]
                    
                    if not is_bbox_fully_visible(bbox, img_w, img_h):
                        skip_sample = True
                        break
                    
                    all_bboxes_orig_raw.append(bbox_raw)
                except Exception as e:
                    print(f"Error processing frame {frame_idx} for sample {i}: {e}")
                    skip_sample = True
                    break
            
            if skip_sample:
                skipped_count += 1
                continue
            
            # --- 创建样本目录 ---
            sample_dir = os.path.join(cropped_images_base_dir, f"sample_{i:06d}")
            os.makedirs(sample_dir, exist_ok=True)
            
            # --- 处理模板帧 ---
            template_indices = list(range(template_frames))
            template_bboxes_orig = [convert_bbox_format(all_bboxes_orig_raw[idx]) for idx in template_indices]
            template_bboxes_jittered = [jitter_bbox(bbox, jitter_scale=0.05) for bbox in template_bboxes_orig]
            
            cropped_template_paths = []
            for t_idx, template_idx in enumerate(template_indices):

                cropped_template = crop_and_pad_template(
                    frame_paths[template_idx], 
                    template_bboxes_jittered[t_idx], 
                    scale=scale, 
                    resize=resize
                )
                template_filename = f"template_{template_idx:03d}.jpg"
                template_save_path = os.path.join(sample_dir, template_filename)
                cropped_template.save(template_save_path)
                cropped_template_paths.append(template_save_path)

            
            if skip_sample:
                skipped_count += 1
                continue
            
            # --- 处理所有搜索帧 (除了模板帧) ---
            search_indices = [idx for idx in range(num_frames) if idx not in template_indices]
            cropped_search_paths = []
            cropped_search_bboxes = []
            skip_sample_due_to_error = False # <--- 新增标志：标记样本是否因错误跳过

            reference_bbox_for_search_crop = template_bboxes_jittered[0]

            for search_idx in search_indices:
                try:
                    search_bbox_orig = convert_bbox_format(all_bboxes_orig_raw[search_idx])

                    cropped_search_img, search_bbox_new, crop_region, _ = crop_and_pad_search(
                        frame_paths[search_idx],
                        reference_bbox_for_search_crop,
                        search_bbox_orig,
                        scale=search_scale,
                        resize=resize
                    )

                    search_filename = f"search_{search_idx:03d}.jpg"
                    search_save_path = os.path.join(sample_dir, search_filename)
                    cropped_search_img.save(search_save_path)

                    cropped_search_paths.append(search_save_path)
                    cropped_search_bboxes.append(search_bbox_new)
                except Exception as e:
                    print(f"Error processing search frame {search_idx} for sample {i}: {e}. Skipping sample.")
                    skip_sample_due_to_error = True # <--- 设置错误标志
                    break # <--- 退出搜索帧循环

            # --- 新增：检查是否因处理错误需要跳过 ---
            if skip_sample_due_to_error:
                try:
                    shutil.rmtree(sample_dir)
                    print(f"Removed directory due to processing error: {sample_dir}")
                except OSError as e:
                    print(f"Error removing directory {sample_dir} after processing error: {e}")
                skipped_count += 1
                continue # <--- 跳到下一个样本

            # --- 修改：检查是否没有成功处理任何搜索帧 ---
            if not cropped_search_paths:
                # 如果没有成功处理任何搜索帧（可能是因为上面break了，或者一开始就没成功）
                print(f"Skipping sample {i} because no search frames were successfully processed.")
                try:
                    # 确保清理目录
                    shutil.rmtree(sample_dir)
                    print(f"Removed directory as no search frames were processed: {sample_dir}")
                except OSError as e:
                    print(f"Error removing directory {sample_dir} when no search frames processed: {e}")
                skipped_count += 1
                continue # <--- 跳到下一个样本



            image_paths_for_sample = cropped_template_paths + cropped_search_paths

            # 构建用户提示
            init_user_content = []

            # 添加模板帧
            for _ in range(template_frames):
                init_user_content.append({"type": "image"})

            # 修改：移除模板帧的边界框信息，只简单描述物体
            template_text = f"\nThese are the template frames showing the object '{exp_str}'."
            init_user_content.append({"text": template_text})

            # 添加所有搜索帧
            for _ in range(len(cropped_search_paths)):
                init_user_content.append({"type": "image"})

            # 添加跟踪指令，使用[x1, y1, x2, y2]格式
            tracking_instruction = f" Please track the object '{exp_str}' in the next frame. "
            tracking_instruction += "provide the bounding box [x1, y1, x2, y2]. "
            tracking_instruction += "Use [0, 0, 0, 0] if the object is not visible."
            init_user_content.append({"text": tracking_instruction})


            init_user_msg = {"role": "user", "content": init_user_content}
            prompt_messages = [init_user_msg]

            # 构建解决方案（仅包含搜索帧的跟踪结果）
            solution = f"Tracking results for '{exp_str}':\n\n"
            any_search_invisible = False # <--- 初始化标志

            # 保持不变：只包含搜索帧结果，直接使用裁剪后的坐标
            temp_solution_lines = [] # <--- 临时存储solution行
            search_indices_processed = search_indices[:len(cropped_search_bboxes)] # 确保索引和bbox对齐
            for idx, (search_idx, search_bbox) in enumerate(zip(search_indices_processed, cropped_search_bboxes)):
                # 检查原始注释中的可见性标志
                # 注意：这里假设 annotations_dict['visible'] 的长度与 num_frames 一致
                # 如果 annotations_dict['visible'] 可能不存在或长度不匹配，需要更健壮的处理
                visibility_list = annotations_dict.get('visible', [True] * num_frames)
                if search_idx >= len(visibility_list):
                     # 处理索引越界的情况，例如将其视为不可见
                     print(f"Warning: search_idx {search_idx} out of bounds for visibility list (len {len(visibility_list)}) for sample {i}. Treating as invisible.")
                     search_visible = False
                else:
                     search_visible = visibility_list[search_idx]


                if search_visible:
                    # 直接使用裁剪后的边界框坐标，不需要normalize
                    temp_solution_lines.append(f"[{int(search_bbox[0])}, {int(search_bbox[1])}, {int(search_bbox[2])}, {int(search_bbox[3])}]\n")
                else:
                    temp_solution_lines.append(f"[0, 0, 0, 0]\n")
                    any_search_invisible = True # <--- 如果有不可见帧，设置标志

            # --- 修改：检查是否有不可见的搜索帧 ---
            if any_search_invisible:
                print(f"Skipping sample {i} because it contains invisible search frames.")
                # 删除已保存的图像和目录
                try:
                    shutil.rmtree(sample_dir)
                    # --- 修改日志 ---
                    print(f"[Invisibility Check] Successfully removed directory: {sample_dir}")
                except OSError as e:
                     # --- 修改日志 ---
                    print(f"[Invisibility Check] Error removing directory {sample_dir}: {e}")
                skipped_count += 1
                continue # <--- 跳到下一个样本

            # --- 如果所有搜索帧都可见，则完成solution字符串 ---
            solution += "".join(temp_solution_lines)

            # 创建样本
            sample_data = {
                "image": image_paths_for_sample,
                "problem": TRACKING_SYSTEM_PROMPT,
                "solution": solution,
                "prompt": prompt_messages
            }

            all_samples.append(sample_data)
            processed_count += 1

        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            # --- 新增：在主异常处理中也尝试删除目录 ---
            # 检查 sample_dir 是否已定义并存在，以防错误发生在创建目录之前
            if 'sample_dir' in locals() and os.path.exists(sample_dir):
                try:
                    shutil.rmtree(sample_dir)
                    print(f"[Outer Exception] Cleaned up directory due to error: {sample_dir}")
                except OSError as remove_error:
                    print(f"[Outer Exception] Error removing directory {sample_dir} after error: {remove_error}")
            skipped_count += 1
            continue
    
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
    print(f"One-turn dataset saved to: {dataset_path}")
    print(f"Total samples attempted: {data_len}")
    print(f"Samples skipped: {skipped_count}")
    print(f"Total conversation samples: {len(all_samples)}")
    print("-" * 30)
    
    return hf_dataset

TRACKING_SYSTEM_PROMPT = (
    "You are a professional visual object tracking assistant. Your task is to track specified target objects in a video sequence. "
    # "The user will provide servelal template frames with the target's bounding box, then you need to find the target's new position in subsequent frames. "
    "The user will provide servelal template frames in the middle of the frames, then you need to find the target's new position in subsequent frames. "
    "Please directly return the target's bounding box coordinates in the format [x1, y1, x2, y2], where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate. "
    "Your answer should be wrapped in <answer>[x1, y1, x2, y2]</answer> tags."
)

if __name__ == "__main__":
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="构建使用裁剪图像的一轮跟踪数据集")
    
    # 添加命令行参数
    parser.add_argument("--samples_per_epoch", type=int, default=20)
    parser.add_argument("--output_dir", default='/data1/lihaobo/tracking/rft/test', type=str)
    parser.add_argument("--template_frames", type=int, default=2)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--search_scale", type=float, default=3.0)
    parser.add_argument("--resize", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    
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
    train_dataset, _ = build_dataset()
    
    # 使用命令行参数设置样本数量
    if args.samples_per_epoch > 0:
        train_dataset.samples_per_epoch = args.samples_per_epoch
        print(f"Processing {args.samples_per_epoch} samples per epoch")
    else:
        print("Processing all samples in the dataset")
    
    # 构建使用裁剪图像的一轮跟踪数据集
    output_dir = args.output_dir
    print(f"Output directory set to: {output_dir}")
    
    build_one_turn_tracking_dataset_cropped(
        train_dataset,
        output_dir=output_dir,
        template_frames=args.template_frames,
        scale=args.scale,
        search_scale=args.search_scale,
        resize=args.resize
    )
    
    # 加载并检查数据集
    from datasets import load_from_disk
    dataset = load_from_disk(f"{output_dir}/tracking_dataset")
    print("Dataset sample:")
    print(dataset[0])
    print(f"Total samples: {len(dataset)}")