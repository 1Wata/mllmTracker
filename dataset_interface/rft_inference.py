import os
import re
import json
import argparse
import logging
import time
import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
import torch.multiprocessing as mp
import math
import functools

# 假设 make_crop_dataset.utils, evaluation.datasets 在 Python 路径中
from make_crop_dataset.utils import convert_bbox_from_cropped_img, crop_and_pad_template, crop_and_pad_search
from evaluation.datasets import get_dataset, SequenceList

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Loading and Utility Functions (No Error Handling) ---

def load_rft_model(model_path, device="cuda"):
    """加载 RFT 微调后的模型 (无错误捕获)"""
    logger.info(f"Loading RFT model from {model_path} onto device {device}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device # 直接指定设备
    )
    logger.info(f"Loading tokenizer and processor from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, tokenizer, processor

def extract_single_bbox(response):
    """从模型响应中提取单个边界框坐标 (无错误捕获)"""
    start_tag = "<answer>"
    end_tag = "</answer>"
    patterns = [
        r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]',
        r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
    ]
    bbox = None
    content_str = None

    if start_tag in response and end_tag in response:
        start_idx = response.find(start_tag) + len(start_tag)
        end_idx = response.find(end_tag)
        if start_idx < end_idx:
            content_str = response[start_idx:end_idx].strip()

    if content_str:
        for pattern in patterns:
            matches = list(re.finditer(pattern, content_str))
            if matches:
                last_match = matches[-1]
                # 尝试浮点数
                try:
                    x1 = round(float(last_match.group(1)), 1)
                    y1 = round(float(last_match.group(2)), 1)
                    x2 = round(float(last_match.group(3)), 1)
                    y2 = round(float(last_match.group(4)), 1)
                    bbox = [x1, y1, x2, y2]
                    break
                except ValueError:
                    # 尝试整数
                    try:
                        x1 = int(last_match.group(1))
                        y1 = int(last_match.group(2))
                        x2 = int(last_match.group(3))
                        y2 = int(last_match.group(4))
                        bbox = [x1, y1, x2, y2]
                        break
                    except ValueError:
                        continue

    if bbox is None: # 标签内未找到，全局查找
        for pattern in patterns:
            matches = list(re.finditer(pattern, response))
            if matches:
                last_match = matches[-1]
                try:
                    x1 = round(float(last_match.group(1)), 1)
                    y1 = round(float(last_match.group(2)), 1)
                    x2 = round(float(last_match.group(3)), 1)
                    y2 = round(float(last_match.group(4)), 1)
                    bbox = [x1, y1, x2, y2]
                    break
                except ValueError:
                    try:
                        x1 = int(last_match.group(1))
                        y1 = int(last_match.group(2))
                        x2 = int(last_match.group(3))
                        y2 = int(last_match.group(4))
                        bbox = [x1, y1, x2, y2]
                        break
                    except ValueError:
                        continue

    if bbox is None:
        logger.warning(f"Failed to extract bounding box from response: {response}")
    return bbox

def draw_bbox(image, bbox, color="red", width=2):
    """在图像上绘制边界框 (无错误捕获)"""
    draw = ImageDraw.Draw(image)
    coords = [(float(bbox[0]), float(bbox[1])), (float(bbox[2]), float(bbox[3]))]
    draw.rectangle(coords, outline=color, width=width)
    return image

def generate_output(model, tokenizer, inputs, max_new_tokens=128):
    """使用模型生成输出 (无错误捕获)"""
    inputs = {k: v.to(model.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    input_token_len = inputs['input_ids'].shape[1]
    generated_part = tokenizer.batch_decode(
        generated_ids[:, input_token_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return generated_part[0] if generated_part else ""

def build_rft_tracking_message(template_imgs, search_img, exp_str):
    """构建符合 RFT 训练格式的测试消息 (无错误捕获)"""
    messages = []
    user_content = []
    if len(template_imgs) > 1:
        user_content.append({"type": "text", "text": f"The first {len(template_imgs)} images show the object of interest: '{exp_str}'. "})
        for _ in template_imgs: user_content.append({"type": "image"})
    elif len(template_imgs) == 1:
        user_content.append({"type": "text", "text": f"This image shows the object of interest: '{exp_str}'. "})
        user_content.append({"type": "image"})
    else:
         raise ValueError("At least one template image is required.")
    user_content.append({"type": "text", "text": "Please locate this object in the final image. Provide its bounding box as [x1, y1, x2, y2] coordinates within that image. Wrap your answer in <answer></answer> tags."})
    user_content.append({"type": "image"})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": ""})
    return messages

# --- Core Tracking Logic (No Error Handling) ---

def process_sequences_rft(model, tokenizer, processor, sequences_to_process, dataset_name,
                           template_crop_scale, search_crop_scale, resize,
                           save_visualize, output_dir, max_new_tokens):
    """
    使用 RFT 模型评估指定序列列表的跟踪性能 (无错误捕获)。
    """
    results = []
    process_times = {
        "image_loading": [], "template_crop": [], "search_crop": [],
        "input_processing": [], "model_inference": [], "output_processing": [],
        "total_frame": []
    }
    vis_dir = None
    if save_visualize:
        vis_dir = os.path.join(output_dir, "visualization")
        # No need to create here, main process or worker can handle it if needed per sequence

    logger.info(f"Processing {len(sequences_to_process)} sequences on device {model.device}")

    for seq in tqdm(sequences_to_process, desc=f"Tracking on {model.device}"):
        seq_results = []
        logger.info(f"Processing sequence: {seq.name} on {model.device}")

        # 获取第一帧信息
        init_info = seq.init_info()
        if not init_info or 'init_bbox' not in init_info or not init_info['init_bbox']:
             logger.error(f"Initial info or bbox missing for sequence {seq.name}. Skipping.")
             continue # Skip sequence if init info is bad
        first_frame_path = seq.frames[0]
        first_frame_bbox = init_info['init_bbox'] # [x, y, w, h]
        exp_str = init_info.get('init_text_description', 'the target object')

        # 创建预测输出文件
        pred_file_path = os.path.join(output_dir, f"{seq.name}.txt")
        with open(pred_file_path, "w") as f:
            x, y, w, h = first_frame_bbox
            f.write(f"{float(x):.2f},{float(y):.2f},{float(w):.2f},{float(h):.2f}\n")

        # 转换 bbox 格式
        first_frame_bbox_xyxy = [
            float(first_frame_bbox[0]), float(first_frame_bbox[1]),
            float(first_frame_bbox[0]) + float(first_frame_bbox[2]),
            float(first_frame_bbox[1]) + float(first_frame_bbox[3])
        ]

        # 创建序列可视化目录 (if needed)
        seq_vis_dir = None
        if save_visualize and vis_dir:
            seq_vis_dir = os.path.join(vis_dir, seq.name)
            os.makedirs(seq_vis_dir, exist_ok=True)

        # 加载和裁剪第一帧
        first_frame = Image.open(first_frame_path).convert("RGB")
        cropped_first_frame = crop_and_pad_template(
            first_frame, first_frame_bbox_xyxy,
            scale=template_crop_scale, resize=resize
        )

        # 初始化追踪状态
        current_bbox_xyxy = first_frame_bbox_xyxy
        current_frame = first_frame

        # 保存第一帧可视化
        if save_visualize and seq_vis_dir:
            frame_0_dir = os.path.join(seq_vis_dir, "frame_0000")
            os.makedirs(frame_0_dir, exist_ok=True)
            first_frame_vis = draw_bbox(first_frame.copy(), first_frame_bbox_xyxy, color="green")
            first_frame_vis.save(os.path.join(frame_0_dir, f"search_original_with_gt_bbox.jpg"))
            cropped_first_frame.save(os.path.join(frame_0_dir, f"template_0000.jpg"))

        # --- 循环处理后续帧 ---
        for i, frame_path in enumerate(seq.frames[1:], start=1):
            frame_start_time = time.time()
            frame_log_data = {'frame_id': i, 'frame_path': frame_path}
            frame_vis_dir = None
            if save_visualize and seq_vis_dir:
                frame_vis_dir = os.path.join(seq_vis_dir, f"frame_{i:04d}")
                os.makedirs(frame_vis_dir, exist_ok=True)

            # 加载当前帧 (搜索帧)
            t_start = time.time()
            search_frame = Image.open(frame_path).convert("RGB")
            process_times["image_loading"].append(time.time() - t_start)
            if save_visualize and frame_vis_dir:
                search_frame.save(os.path.join(frame_vis_dir, "search_original.jpg"))

            # 准备模板帧
            t_start_template = time.time()
            cropped_templates = [cropped_first_frame]
            cropped_current_template = crop_and_pad_template(
                current_frame, current_bbox_xyxy,
                scale=template_crop_scale, resize=resize
            )
            cropped_templates.append(cropped_current_template)
            process_times["template_crop"].append(time.time() - t_start_template)

            if save_visualize and frame_vis_dir:
                if len(cropped_templates) > 0: cropped_templates[0].save(os.path.join(frame_vis_dir, "template_first.jpg"))
                if len(cropped_templates) > 1: cropped_templates[1].save(os.path.join(frame_vis_dir, "template_last.jpg"))

            # 裁剪搜索帧
            t_start_search = time.time()
            cropped_search_img, _, crop_region, _ = crop_and_pad_search(
                search_frame, current_bbox_xyxy, None,
                scale=search_crop_scale, resize=resize
            )
            process_times["search_crop"].append(time.time() - t_start_search)
            if save_visualize and frame_vis_dir:
                cropped_search_img.save(os.path.join(frame_vis_dir, "search_cropped.jpg"))

            # 构建消息和处理输入
            t_start_proc = time.time()
            messages = build_rft_tracking_message(cropped_templates, cropped_search_img, exp_str)
            model_input_images = cropped_templates + [cropped_search_img]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=model_input_images, return_tensors="pt", padding=True)
            process_times["input_processing"].append(time.time() - t_start_proc)

            if save_visualize and frame_vis_dir:
                frame_log_data['input_prompt_text'] = text

            # 生成输出
            t_start_inference = time.time()
            response = generate_output(model, tokenizer, inputs, max_new_tokens)
            process_times["model_inference"].append(time.time() - t_start_inference)

            if save_visualize and frame_vis_dir:
                frame_log_data['model_response'] = response

            # 提取边界框并处理
            t_start_output_proc = time.time()
            predicted_bbox = extract_single_bbox(response)
            abs_bbox = None

            if predicted_bbox:
                 frame_log_data['predicted_bbox_cropped'] = predicted_bbox
                 if save_visualize and frame_vis_dir and cropped_search_img:
                    vis_cropped = draw_bbox(cropped_search_img.copy(), predicted_bbox, color="blue")
                    vis_cropped.save(os.path.join(frame_vis_dir, "search_cropped_with_pred_bbox.jpg"))

                 if crop_region:
                    abs_bbox = convert_bbox_from_cropped_img(crop_region, predicted_bbox, resize)
                    frame_log_data['predicted_bbox_original'] = abs_bbox

                    current_frame = search_frame
                    current_bbox_xyxy = abs_bbox

                    x, y, w, h = abs_bbox[0], abs_bbox[1], abs_bbox[2]-abs_bbox[0], abs_bbox[3]-abs_bbox[1]
                    with open(pred_file_path, "a") as f:
                        f.write(f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n")
                    seq_results.append({'frame_id': i, 'bbox': [x, y, w, h], 'status': 'success'})
                    frame_log_data['status'] = 'success'

                    if save_visualize and frame_vis_dir:
                        search_vis = draw_bbox(search_frame.copy(), abs_bbox, color="red")
                        search_vis.save(os.path.join(frame_vis_dir, "search_original_with_pred_bbox.jpg"))
                 else: # crop_region is None (Should not happen if cropping succeeded)
                    logger.warning(f"Predicted bbox found but no crop_region for frame {i}, seq {seq.name}. Writing previous bbox.")
                    x, y, w, h = current_bbox_xyxy[0], current_bbox_xyxy[1], current_bbox_xyxy[2]-current_bbox_xyxy[0], current_bbox_xyxy[3]-current_bbox_xyxy[1]
                    with open(pred_file_path, "a") as f: f.write(f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n")
                    seq_results.append({'frame_id': i, 'bbox': [x, y, w, h], 'status': 'no_crop_region_post_pred'})
                    frame_log_data['status'] = 'no_crop_region_post_pred'
            else:
                # 提取边界框失败
                logger.warning(f"Failed to extract bbox from response for frame {i} in sequence {seq.name}. Writing previous bbox.")
                x, y, w, h = current_bbox_xyxy[0], current_bbox_xyxy[1], current_bbox_xyxy[2]-current_bbox_xyxy[0], current_bbox_xyxy[3]-current_bbox_xyxy[1]
                with open(pred_file_path, "a") as f: f.write(f"{x:.2f},{y:.2f},{w:.2f},{h:.2f}\n")
                seq_results.append({'frame_id': i, 'bbox': [x, y, w, h], 'status': 'extraction_failed'})
                frame_log_data['status'] = 'extraction_failed'

            process_times["output_processing"].append(time.time() - t_start_output_proc)

            # 保存日志
            if save_visualize and frame_vis_dir:
                 log_file_path = os.path.join(frame_vis_dir, "log_data.json")
                 def default_serializer(obj):
                     if isinstance(obj, (np.ndarray, Image.Image)): return str(obj)
                     elif isinstance(obj, (np.float32, np.float64)): return float(obj)
                     elif isinstance(obj, (np.int32, np.int64)): return int(obj)
                     raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
                 with open(log_file_path, "w") as f_log:
                     json.dump(frame_log_data, f_log, indent=2, default=default_serializer)

            process_times["total_frame"].append(time.time() - frame_start_time)
            # --- Frame loop end ---

        results.append({'sequence_name': seq.name, 'frames': seq_results})
        logger.info(f"Finished sequence: {seq.name} on {model.device}")
        # --- Sequence loop end ---

    return process_times, results

# --- Multiprocessing Worker ---

def run_worker(rank, world_size, args, sequences_chunk):
    """
    Worker function for multiprocessing. Loads model and processes a chunk of sequences.
    """
    device = f"cuda:{rank}"
    logger.info(f"Worker {rank}/{world_size} starting on device {device}")

    # 每个 worker 加载自己的模型实例
    model, tokenizer, processor = load_rft_model(args.model_path, device=device)

    # 确定输出目录 (所有 worker 写入同一个主目录)
    if args.output_dir is None:
        model_name = os.path.basename(args.model_path.rstrip('/')) or "rft_model"
        output_dir = f"tracking_results_rft/{args.dataset_name}_{model_name}"
    else:
        output_dir = args.output_dir
    # 主进程已创建 output_dir

    # 处理分配给该 worker 的序列
    process_times, results = process_sequences_rft(
        model, tokenizer, processor, sequences_chunk, args.dataset_name,
        args.template_scale, args.search_scale, args.resize,
        args.save_vis, output_dir, args.max_new_tokens
    )

    logger.info(f"Worker {rank}/{world_size} finished.")
    # 返回性能统计信息
    return process_times

# --- Main Function ---

def main():
    parser = argparse.ArgumentParser(description="Run RFT Tracking Model Inference with Multiprocessing")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the RFT fine-tuned model directory")
    parser.add_argument("--dataset_name", type=str, default="OTB_lang",
                        help="Dataset name for evaluation")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results. If None, defaults to 'tracking_results_rft/[dataset_name]_[model_name]'")
    parser.add_argument("--sequence", type=str, default=None, nargs='+',
                        help="Specific sequence(s) to test. If None, all sequences are processed.")
    parser.add_argument("--save_vis", action="store_true", default=False,
                        help="Save visualization results")
    parser.add_argument("--template_scale", type=float, default=2.0,
                        help="Scale factor for template cropping")
    parser.add_argument("--search_scale", type=float, default=4.0,
                        help="Scale factor for search region cropping")
    parser.add_argument("--resize", type=int, default=320,
                        help="Size to resize cropped images")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Maximum new tokens for the model to generate")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes. Defaults to number of GPUs.")

    args = parser.parse_args()

    # --- Dataset Loading and Filtering ---
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset = get_dataset(args.dataset_name)
    if not dataset:
        logger.error(f"Failed to load dataset: {args.dataset_name}")
        exit(1)

    if args.sequence:
        filtered_dataset = []
        sequence_names_in_dataset = [seq.name for seq in dataset]
        for seq_name in args.sequence:
            if seq_name in sequence_names_in_dataset:
                for seq in dataset:
                    if seq.name == seq_name:
                        filtered_dataset.append(seq)
                        break
            else:
                 logger.warning(f"Sequence '{seq_name}' not found in dataset {args.dataset_name}")
        if not filtered_dataset:
            logger.error(f"None of the specified sequences {args.sequence} found in dataset {args.dataset_name}. Exiting.")
            exit(1)
        sequences_to_run = SequenceList(filtered_dataset)
    else:
        sequences_to_run = dataset

    logger.info(f"Total sequences to process: {len(sequences_to_run)}")

    # --- Output Directory Setup ---
    if args.output_dir is None:
        model_name = os.path.basename(args.model_path.rstrip('/')) or "rft_model"
        args.output_dir = f"tracking_results_rft/{args.dataset_name}_{model_name}"
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Output directory: {args.output_dir}")
    if args.save_vis:
         vis_dir = os.path.join(args.output_dir, "visualization")
         os.makedirs(vis_dir, exist_ok=True)
         logger.info(f"Visualization directory: {vis_dir}")


    # --- Multiprocessing Setup ---
    world_size = args.num_workers if args.num_workers else torch.cuda.device_count()
    if world_size == 0:
        logger.error("No CUDA devices found. Multiprocessing requires GPUs.")
        exit(1)
    if world_size > torch.cuda.device_count():
        logger.warning(f"Requested {world_size} workers, but only {torch.cuda.device_count()} GPUs available. Using {torch.cuda.device_count()} workers.")
        world_size = torch.cuda.device_count()
    if len(sequences_to_run) < world_size:
        logger.warning(f"Number of sequences ({len(sequences_to_run)}) is less than number of workers ({world_size}). Reducing workers to {len(sequences_to_run)}.")
        world_size = len(sequences_to_run)

    logger.info(f"Starting multiprocessing with {world_size} workers.")

    sequences_list = list(sequences_to_run) # Convert SequenceList to list for easy slicing
    chunk_size = math.ceil(len(sequences_list) / world_size)
    chunks = [sequences_list[i:i + chunk_size] for i in range(0, len(sequences_list), chunk_size)]

    # Ensure we have exactly world_size chunks if possible, handle edge cases
    if len(chunks) > world_size:
         # This can happen if chunk_size calculation leads to one extra small chunk
         last_chunk = chunks.pop()
         chunks[-1].extend(last_chunk)
    while len(chunks) < world_size: # Should not happen with ceil, but as safeguard
        chunks.append([])

    # Use mp.spawn to start processes
    worker_func = functools.partial(run_worker, world_size=world_size, args=args)
    ctx = mp.get_context('spawn') # Use spawn context for CUDA safety
    with ctx.Pool(processes=world_size) as pool:
         # Map chunks to workers
         all_process_times = pool.map(worker_func, [(rank, chunks[rank]) for rank in range(world_size)])


    # --- Aggregate Results ---
    logger.info("Aggregating results from workers...")
    aggregated_times = {
        "image_loading": [], "template_crop": [], "search_crop": [],
        "input_processing": [], "model_inference": [], "output_processing": [],
        "total_frame": []
    }
    for worker_times in all_process_times:
        if worker_times: # Check if worker returned valid times
            for key in aggregated_times:
                if key in worker_times:
                    aggregated_times[key].extend(worker_times[key])

    # --- Final Logging ---
    logger.info("--- Overall Performance Stats ---")
    for key, times in aggregated_times.items():
        if times:
            avg_time = sum(times) / len(times)
            logger.info(f"Average {key} time: {avg_time:.4f}s (across {len(times)} frames)")
        else:
            logger.info(f"No data for {key} time.")
    logger.info("-------------------------------")
    logger.info(f"Evaluation complete. Bbox results saved to {args.output_dir}")
    # Note: Overall results JSON is not generated in this MP version, focus is on bbox files.

if __name__ == "__main__":
    # Set start method for multiprocessing (important for CUDA)
    # mp.set_start_method('spawn', force=True) # Set globally if needed, or use context as above
    main()