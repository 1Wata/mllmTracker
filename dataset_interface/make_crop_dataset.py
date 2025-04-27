import os
from PIL import Image, ImageDraw # Import ImageDraw
import json
from tqdm import tqdm
import numpy as np
import random
import torch
from build_pytorch_dataset import build_dataset
# 确保导入了必要的工具函数
from make_crop_dataset.utils import crop_and_pad_template, crop_and_pad_search, jitter_bbox, convert_bbox_format

# 移除 normalize_bbox_xyhw 函数，因为它不再需要

# --- Helper function for visibility check ---
def is_bbox_fully_visible(bbox, img_width, img_height):
    """Checks if the bounding box [x1, y1, x2, y2] is fully within image bounds."""
    # Basic check for valid bbox format first
    if not (bbox and len(bbox) == 4):
        return False
    x1, y1, x2, y2 = bbox
    # Check for valid coordinates and dimensions
    if x1 >= x2 or y1 >= y2:
        return False
    # Check against image boundaries
    return x1 >= 0 and y1 >= 0 and x2 <= img_width and y2 <= img_height

def build_json_dataset(pytorch_dataset, output_dir="output_datasets", template_frames=1, scale=2.0, search_scale=4.0, resize=320):
    """
    裁剪图像并构建类似 ShareGPT 格式的数据集 (messages, images)。
    支持多个模板帧和单个搜索帧。使用绝对图像路径。
    仅在所有可见性检查通过（包括目标在搜索裁剪区域内）后才保存图像。

    Args:
        pytorch_dataset: PyTorch 数据集实例。
        output_dir: 输出目录。
        template_frames: 使用的模板帧数量。
        scale: 模板帧裁剪的缩放因子。
        search_scale: 搜索帧裁剪的缩放因子。
        resize: 裁剪后图像的尺寸 (所有裁剪后的图像都将是 resize x resize)。
    """
    # Ensure output_dir is absolute
    output_dir = os.path.abspath(output_dir)
    cropped_images_base_dir = os.path.join(output_dir, "cropped_images")
    os.makedirs(cropped_images_base_dir, exist_ok=True)

    # 用于存储最终JSON数据
    all_data = []

    skipped_count = 0
    processed_count = 0

    # Check if samples_per_epoch is set and use it, otherwise use full dataset length
    if hasattr(pytorch_dataset, 'samples_per_epoch') and pytorch_dataset.samples_per_epoch is not None:
        data_len = min(len(pytorch_dataset), pytorch_dataset.samples_per_epoch)
        print(f"Processing {data_len} samples based on samples_per_epoch setting.")
    else:
        data_len = len(pytorch_dataset)
        print(f"Processing all {data_len} samples in the dataset.")


    for i in tqdm(range(data_len), desc="Processing sequences and building JSON"):
        try:
            sample = pytorch_dataset[i]

            frame_paths = sample.get("images")
            annotations_dict = sample.get("anno")
            dataset_name = sample.get("dataset_name", "unknown")
            exp_str = sample.get("exp_str", f"sequence_{i}")
            if exp_str.endswith((" ", "\n", ".")):
                exp_str = exp_str[:-1]

            # 检查序列是否有足够的帧
            if len(frame_paths) < template_frames + 1:
                skipped_count += 1
                continue

            # --- 选择帧 ---
            template_indices = list(range(template_frames))
            search_idx = min(len(frame_paths) - 1, random.randint(template_frames, template_frames + 50))

            # --- 获取原始数据 (路径和BBox) ---
            template_paths = [frame_paths[idx] for idx in template_indices]
            template_bboxes_orig_raw = [annotations_dict["bbox"][idx].tolist() if isinstance(annotations_dict["bbox"][idx], torch.Tensor) else annotations_dict["bbox"][idx] for idx in template_indices]
            template_bboxes_orig = [convert_bbox_format(bbox) for bbox in template_bboxes_orig_raw] # List of [x1, y1, x2, y2]

            search_path = frame_paths[search_idx]
            search_bbox_orig_raw = annotations_dict["bbox"][search_idx].tolist() if isinstance(annotations_dict["bbox"][search_idx], torch.Tensor) else annotations_dict["bbox"][search_idx]
            search_bbox_orig = convert_bbox_format(search_bbox_orig_raw) # [x1, y1, x2, y2]

            # --- 执行原始图像可见性检查 ---
            skip_sample = False
            # 检查模板帧
            for t_idx, template_idx_val in enumerate(template_indices):
                try:
                    with Image.open(template_paths[t_idx]) as img:
                        t_w, t_h = img.size
                    if not is_bbox_fully_visible(template_bboxes_orig[t_idx], t_w, t_h):
                        skip_sample = True
                        break
                except Exception as img_err:
                    print(f"Warning: Error reading template image {template_paths[t_idx]} for sample {i}: {img_err}")
                    skip_sample = True
                    break
            if skip_sample:
                skipped_count += 1
                continue

            # 检查搜索帧
            try:
                with Image.open(search_path) as img:
                    s_w, s_h = img.size
                if not is_bbox_fully_visible(search_bbox_orig, s_w, s_h):
                    skip_sample = True
            except Exception as img_err:
                print(f"Warning: Error reading search image {search_path} for sample {i}: {img_err}")
                skip_sample = True

            if skip_sample:
                skipped_count += 1
                continue

            # --- 应用抖动到模板BBox ---
            template_bboxes_jittered = [jitter_bbox(bbox, jitter_scale=0.05) for bbox in template_bboxes_orig]

            # --- 尝试裁剪搜索帧并检查目标是否在裁剪区域内 ---
            # 使用第一个抖动后的模板框作为参考来确定搜索区域
            reference_bbox_for_search_crop = template_bboxes_jittered[0]
            try:
                # 调用 crop_and_pad_search，如果目标不在区域内会抛出 ValueError
                cropped_search_img, search_bbox_new, crop_region, _ = crop_and_pad_search(
                    search_path, reference_bbox_for_search_crop, search_bbox_orig,
                    scale=search_scale, resize=resize
                )
                # 如果到这里没有出错，说明目标在搜索裁剪区域内

            except ValueError as e: # 目标不在裁剪区域内
                # print(f"Skipping sample {i} because target is outside crop region during search crop: {e}")
                skipped_count += 1
                continue # 跳过此样本，不保存任何内容
            except Exception as crop_err: # 其他裁剪错误
                print(f"Error during initial search crop check for sample {i}: {crop_err}")
                skipped_count += 1
                continue # 跳过此样本

            # --- 所有检查通过，现在创建目录并处理/保存图像 ---
            sample_dir = os.path.join(cropped_images_base_dir, f"sample_{i:06d}")
            os.makedirs(sample_dir, exist_ok=True)

            # --- 处理并保存模板帧 ---
            cropped_template_abs_paths = [] # Store absolute paths
            skip_due_to_template_error = False
            for t_idx, template_idx_val in enumerate(template_indices):
                t_path = template_paths[t_idx]
                t_bbox_jittered = template_bboxes_jittered[t_idx] # 使用之前抖动过的bbox

                try:
                    cropped_template = crop_and_pad_template(t_path, t_bbox_jittered, scale=scale, resize=resize)
                    template_filename = f"template_{template_idx_val:03d}.jpg"
                    template_save_path = os.path.join(sample_dir, template_filename) # Absolute path
                    cropped_template.save(template_save_path)
                    cropped_template_abs_paths.append(template_save_path) # Store absolute path
                except Exception as crop_err:
                    print(f"Error cropping/saving template {template_idx_val} for sample {i} (after passing checks): {crop_err}")
                    skip_due_to_template_error = True # 标记错误
                    break # 停止处理此样本的后续模板
            
            if skip_due_to_template_error:
                # 如果模板裁剪/保存失败，清理已创建的目录和文件
                import shutil
                shutil.rmtree(sample_dir, ignore_errors=True)
                skipped_count += 1
                continue

            # --- 保存之前已成功裁剪的搜索帧 ---
            try:
                search_filename = "search.jpg"
                search_save_path = os.path.join(sample_dir, search_filename) # Absolute path
                cropped_search_img.save(search_save_path) # cropped_search_img 来自之前的 try 块
                search_image_abs_path = search_save_path # Store absolute path
            except Exception as save_err:
                 print(f"Error saving search frame for sample {i} (after passing checks): {save_err}")
                 # 清理已创建的目录和文件
                 import shutil
                 shutil.rmtree(sample_dir, ignore_errors=True)
                 skipped_count += 1
                 continue

            # --- 构建 messages 和 images ---
            messages = []
            image_paths_for_json = cropped_template_abs_paths + [search_image_abs_path] # Use absolute paths
            image_tags = "".join([f"<image>" for _ in image_paths_for_json])

            if template_frames == 1:
                user_content = (f"This image ({image_tags[0:7]}) shows the object of interest: '{exp_str}'. "
                                f"Please locate this object in the following image ({image_tags[7:14]})."
                                f" Provide its bounding box as [x1, y1, x2, y2] coordinates within that image.")
            else:
                 user_content = (f"The first {template_frames} images ({image_tags[0:template_frames*7]}) show the object of interest: '{exp_str}'. "
                                 f"Please locate this object in the final image ({image_tags[template_frames*7:(template_frames+1)*7]})."
                                 f" Provide its bounding box as [x1, y1, x2, y2] coordinates within that image.")

            messages.append({"role": "user", "content": user_content})
            formatted_bbox = [int(coord) for coord in search_bbox_new] # search_bbox_new 来自之前的 try 块
            assistant_content = f"The object '{exp_str}' is located at [{formatted_bbox[0]}, {formatted_bbox[1]}, {formatted_bbox[2]}, {formatted_bbox[3]}]."
            messages.append({"role": "assistant", "content": assistant_content})

            # --- 组合最终的JSON条目 ---
            entry = {
                "id": f"sample_{i:06d}",
                "messages": messages,
                "images": image_paths_for_json
            }
            all_data.append(entry)
            processed_count += 1

        except Exception as e:
            print(f"Major error processing sample {i}: {e}")
            import traceback
            traceback.print_exc() # 打印详细错误堆栈
            skipped_count += 1
            # 尝试清理可能已创建的目录 (如果错误发生在目录创建之后)
            sample_dir_potential = os.path.join(cropped_images_base_dir, f"sample_{i:06d}")
            if os.path.exists(sample_dir_potential):
                 import shutil
                 shutil.rmtree(sample_dir_potential, ignore_errors=True)
            continue

    # --- 保存最终的JSON文件 ---
    json_filename = "sharegpt_cropped_tracking_dataset_abs_path_visible.json"
    output_json_path = os.path.join(output_dir, json_filename)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    # --- 打印统计信息 ---
    print("-" * 50)
    print(f"ShareGPT-like dataset saved to: {output_json_path}")
    print(f"Cropped images saved in: {cropped_images_base_dir}")
    print(f"Total sequences attempted: {data_len}")
    print(f"Sequences skipped due to errors or target visibility: {skipped_count}")
    print(f"Samples included in JSON: {processed_count}")
    print("-" * 50)

    return output_json_path, cropped_images_base_dir


if __name__ == "__main__":
    # 设置随机种子
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    print(f"Random seed set to: {seed}")

    # 构建PyTorch数据集
    train_dataset, _ = build_dataset()

    # 设置处理的样本数量 (可选, 用于测试)

    train_dataset.samples_per_epoch = 20000


    # 构建数据集
    output_dir = "/data1/lihaobo/tracking/data/cropped_sft"
    json_path, images_dir = build_json_dataset(
        train_dataset,
        output_dir=output_dir,
        template_frames=2,
        scale=2.0,
        search_scale=4.0,
        resize=320
    )

    print(f"Dataset creation complete, JSON file: {json_path}")
    print(f"Image directory: {images_dir}")