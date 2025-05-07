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
import matplotlib.pyplot as plt
from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
# from qwen_vl_utils import process_vision_info # 不再需要

from open_r1.utils.utils import transform_bbox, smart_resize


import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
)
print(sys.path)
from dataset_interface.make_crop_dataset.utils import convert_bbox_format
from dataset_interface.evaluation.datasets import get_dataset, SequenceList

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, device="cuda"):
    """加载全量微调的模型"""
    logger.info(f"Loading fully fine-tuned model from {model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device
    )
    logger.info("Model loaded.")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model.eval()
    return model, tokenizer, processor

def extract_bbox_from_response(response):
    """从模型响应中提取边界框坐标，支持<answer>标签格式"""
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    content = answer_match.group(1).strip() if answer_match else response

    pattern = r'\[(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+)\]'
    matches = list(re.finditer(pattern, content))

    if matches:
        last_match = matches[-1]
        # 直接返回浮点数，后续转换时再处理精度
        x1 = float(last_match.group(1))
        y1 = float(last_match.group(2))
        x2 = float(last_match.group(3))
        y2 = float(last_match.group(4))
        return [x1, y1, x2, y2]

    logger.warning(f"Failed to extract bounding box from response")
    return None

def draw_bbox(image, bbox, color="red", width=2):
    """在图像上绘制边界框"""
    draw = ImageDraw.Draw(image)
    # 确保坐标是整数或浮点数
    try:
        coords = [(float(bbox[0]), float(bbox[1])), (float(bbox[2]), float(bbox[3]))]
        draw.rectangle(coords, outline=color, width=width)
    except (ValueError, TypeError) as e:
        logger.error(f"Error drawing bbox {bbox}: {e}")
    return image

def generate_output(model, tokenizer, inputs, max_new_tokens=2048):
    """使用模型生成输出"""
    inputs = inputs.to(model.device)
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return response[0]

def build_test_message(template_imgs, search_img, template_bboxes_orig, exp_str):
    """
    构建测试消息 - 使用原始图像，并在文本中包含模板帧的原始边界框

    Args:
        template_imgs: 模板图像列表 (PIL Image 对象)
        search_img: 搜索图像 (PIL Image 对象)
        template_bboxes_orig: 模板图像中的原始边界框列表 [[x1, y1, x2, y2], ...]
        exp_str: 目标描述

    Returns:
        构建好的消息列表
    """
    messages = []
    num_templates = len(template_imgs)
    template_image_tokens = "".join(["<image>"] * num_templates)

    # 构建用户内容
    user_content_parts = []

    # 添加模板图像占位符
    for _ in range(num_templates):
        user_content_parts.append({"type": "image"})

    # 添加模板图像描述和边界框信息
    template_intro = f"The first {num_templates} image{'s' if num_templates > 1 else ''} ({template_image_tokens}) show the object of interest: '{exp_str}'."
    user_content_parts.append({"text": template_intro})

    # 添加每个模板帧的原始边界框 (与 make_rft_dataset.py 的 no_crop 模式对齐)
    for idx, bbox_orig in enumerate(template_bboxes_orig):
        if bbox_orig:
            bbox_text = f" The bounding box for template frame {idx+1} is: [{int(bbox_orig[0])}, {int(bbox_orig[1])}, {int(bbox_orig[2])}, {int(bbox_orig[3])}]."
            user_content_parts.append({"text": bbox_text})

    # 添加搜索图像占位符
    user_content_parts.append({"type": "image"})

    # 添加定位指令
    locate_instruction = f" Please locate this object in the final image (<image>). Provide its bounding box as [x1, y1, x2, y2] coordinates within that image."
    user_content_parts.append({"text": locate_instruction})

    messages.append({"role": "user", "content": user_content_parts})
    return messages


def evaluate_tracking(model, tokenizer, processor, dataset_name="lasot",
                      sequences=None, save_visualize=False, output_dir=None, max_new_tokens=2048):
    """
    使用原始图像评估跟踪性能 (无裁剪)
    """
    if output_dir is None:
        output_dir = f"tracking_results_no_crop/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)

    vis_dir = None
    if save_visualize:
        vis_dir = os.path.join(output_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)

    dataset = get_dataset(dataset_name)
    results = []

    if sequences:
        filtered_dataset = [seq for seq_name in sequences for seq in dataset if seq.name == seq_name]
        if not filtered_dataset:
            logger.warning(f"Specified sequences not found in dataset {dataset_name}")
            return []
        dataset = SequenceList(filtered_dataset)

    logger.info(f"Processing {len(dataset)} sequences from {dataset_name} (No Cropping)")

    process_times = {
        "image_loading": [], "size_calculation": [], "input_processing": [],
        "model_inference": [], "bbox_extraction": [], "bbox_conversion": [],
        "total_frame": []
    }

    # 获取处理器的图像处理参数 (用于 smart_resize)
    image_processor = processor.image_processor
    min_pixels = getattr(image_processor, 'min_pixels', 56 * 56)
    max_pixels = getattr(image_processor, 'max_pixels', 14 * 14 * 4 * 1280)
    patch_size = getattr(image_processor, 'patch_size', 14) # 假设默认值
    merge_size = getattr(image_processor, 'merge_size', 2) # 假设默认值
    factor = patch_size * merge_size

    for seq in tqdm(dataset, desc="Tracking progress (No Crop)"):
        seq_results = []
        init_info = seq.init_info()
        first_frame_path = seq.frames[0]
        first_frame_bbox = init_info.get('init_bbox') # [x, y, w, h]
        exp_str = init_info.get('init_text_description')

        pred_file_path = os.path.join(output_dir, f"{seq.name}.txt")
        with open(pred_file_path, "w") as f:
            f.write(f"{first_frame_bbox[0]},{first_frame_bbox[1]},{first_frame_bbox[2]},{first_frame_bbox[3]}\n")

        first_frame_bbox_xyxy = convert_bbox_format(first_frame_bbox) # [x1, y1, x2, y2]

        seq_vis_dir = None
        if save_visualize and vis_dir:
            seq_vis_dir = os.path.join(vis_dir, seq.name)
            os.makedirs(seq_vis_dir, exist_ok=True)

        # --- 加载第一帧 ---
        first_frame = Image.open(first_frame_path).convert("RGB")

        # --- 初始化状态 ---
        # current_bbox_xyxy 存储的是 *原始* 坐标系的 bbox
        current_bbox_xyxy = first_frame_bbox_xyxy
        # current_frame 存储的是上一帧的 *原始* PIL Image 对象
        current_frame = first_frame

        # 保存第一帧可视化
        if save_visualize and seq_vis_dir:
            frame_0_dir = os.path.join(seq_vis_dir, "frame_0000")
            os.makedirs(frame_0_dir, exist_ok=True)
            first_frame_vis = draw_bbox(first_frame.copy(), first_frame_bbox_xyxy, color="green") # GT用绿色
            first_frame_vis.save(os.path.join(frame_0_dir, f"original_frame_with_gt_bbox.jpg"))
            # 不再保存裁剪的模板

        # --- 循环处理后续帧 ---
        for i, frame_path in enumerate(seq.frames[1:], start=1):
            frame_start_time = time.time()
            frame_log_data = {}
            frame_vis_dir = None
            if save_visualize and seq_vis_dir:
                frame_vis_dir = os.path.join(seq_vis_dir, f"frame_{i:04d}")
                os.makedirs(frame_vis_dir, exist_ok=True)

            # --- 加载搜索帧 ---
            t_start = time.time()
            search_frame = Image.open(frame_path).convert("RGB")
            process_times["image_loading"].append(time.time() - t_start)
            if save_visualize and frame_vis_dir:
                search_frame.save(os.path.join(frame_vis_dir, "search_original.jpg"))

            # --- 准备模板帧 (原始图像) ---
            # 使用第一帧和上一帧的原始图像
            template_imgs = [first_frame, current_frame]
            # 对应的原始边界框
            template_bboxes_orig = [first_frame_bbox_xyxy, current_bbox_xyxy]

            # --- 计算尺寸信息 (用于坐标转换) ---
            t_start_size = time.time()
            # 需要搜索帧的原始尺寸和预期的调整后尺寸
            search_orig_w, search_orig_h = search_frame.size
            original_size = (search_orig_w, search_orig_h)
            try:
                resized_h, resized_w = smart_resize(
                    search_orig_h, search_orig_w, factor, min_pixels, max_pixels
                )
                resized_size = (resized_w, resized_h) # (W, H)
            except ValueError as e:
                 logger.error(f"Error calculating resized dimensions for frame {i}, seq {seq.name}: {e}. Skipping frame.")
                 # 写入上一帧结果并跳过
                 with open(pred_file_path, "a") as f:
                     prev_x, prev_y = current_bbox_xyxy[0], current_bbox_xyxy[1]
                     prev_w, prev_h = current_bbox_xyxy[2]-prev_x, current_bbox_xyxy[3]-prev_y
                     f.write(f"{prev_x},{prev_y},{prev_w},{prev_h}\n")
                 seq_results.append({'frame_id': i, 'bbox': [prev_x, prev_y, prev_w, prev_h]})
                 if save_visualize and frame_vis_dir: frame_log_data['status'] = 'size_calculation_error'
                 continue # 跳到下一帧

            process_times["size_calculation"].append(time.time() - t_start_size)
            if save_visualize and frame_vis_dir:
                frame_log_data['original_size'] = original_size
                frame_log_data['resized_size_calculated'] = resized_size


            # --- 构建消息和处理输入 ---
            t_start_processing = time.time()
            # 使用原始模板边界框构建消息
            messages = build_test_message(
                template_imgs, search_frame, template_bboxes_orig, exp_str
            )
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # 传递原始图像列表给处理器
            model_input_images = template_imgs + [search_frame]

            # 直接使用 processor 处理文本和图像列表
            # 假设 processor 内部会进行类似 smart_resize 的处理
            inputs = processor(
                text=[text],
                images=model_input_images,
                return_tensors="pt",
                padding=True,
            )
            process_times["input_processing"].append(time.time() - t_start_processing)
            if save_visualize and frame_vis_dir:
                frame_log_data['input_prompt'] = text
                # 可以选择性地保存处理器调整大小后的图像（如果能获取的话）
                # processed_img = processor.image_processor.post_process_image(inputs['pixel_values'][0]) # 示例，具体方法需查阅文档
                # if processed_img: processed_img.save(os.path.join(frame_vis_dir, "search_processed_by_processor.jpg"))


            # --- 模型推理 ---
            t_start_inference = time.time()
            response = generate_output(model, tokenizer, inputs, max_new_tokens)
            process_times["model_inference"].append(time.time() - t_start_inference)
            if save_visualize and frame_vis_dir:
                frame_log_data['model_response'] = response

            # --- 提取和转换边界框 ---
            t_start_extraction = time.time()
            # 模型输出的 bbox 是在 *处理器调整后* 的坐标系中
            predicted_bbox_resized = extract_bbox_from_response(response)
            process_times["bbox_extraction"].append(time.time() - t_start_extraction)

            abs_bbox = None # 初始化绝对坐标 (原始坐标系)
            if predicted_bbox_resized:
                if save_visualize and frame_vis_dir:
                    frame_log_data['predicted_bbox_resized'] = predicted_bbox_resized
                    # 可视化 resized bbox (需要获取 processor resize后的图像，比较困难)
                    # vis_resized = draw_bbox(processed_img.copy(), predicted_bbox_resized, color="blue")
                    # vis_resized.save(os.path.join(frame_vis_dir, "search_processed_with_pred_bbox.jpg"))

                # --- 核心转换：将 resized bbox 转换回 original bbox ---
                t_start_conversion = time.time()
                abs_bbox = transform_bbox(predicted_bbox_resized, original_size, resized_size, 'resized_to_original')
                process_times["bbox_conversion"].append(time.time() - t_start_conversion)

                if abs_bbox:
                    if save_visualize and frame_vis_dir:
                        frame_log_data['predicted_bbox_original'] = abs_bbox
                        # 在原始搜索帧上绘制转换后的 bbox
                        search_vis = draw_bbox(search_frame.copy(), abs_bbox, color="red")
                        search_vis.save(os.path.join(frame_vis_dir, "search_original_with_pred_bbox.jpg"))
                else:
                    # 转换失败
                    logger.warning(f"Bbox conversion failed for frame {i}, seq {seq.name}. Predicted(resized): {predicted_bbox_resized}, OrigSize: {original_size}, ResizedSize: {resized_size}")
                    if save_visualize and frame_vis_dir: frame_log_data['status'] = 'conversion_failed'

            else:
                # 提取失败
                process_times["bbox_conversion"].append(time.time() - t_start_extraction) # 记录失败尝试的时间
                logger.warning(f"Bbox extraction failed for frame {i}, seq {seq.name}.")
                if save_visualize and frame_vis_dir: frame_log_data['status'] = 'extraction_failed'


            # --- 更新状态和保存结果 ---
            if abs_bbox:
                # 成功预测和转换，更新状态
                current_frame = search_frame.copy() # 更新为当前帧 (原始图像)
                current_bbox_xyxy = abs_bbox      # 更新为当前预测的绝对坐标 (原始坐标系)

                # 转换为 [x, y, w, h] 格式保存
                x, y = abs_bbox[0], abs_bbox[1]
                w, h = abs_bbox[2] - x, abs_bbox[3] - y

                with open(pred_file_path, "a") as f:
                    f.write(f"{x},{y},{w},{h}\n")
                seq_results.append({'frame_id': i, 'bbox': [x, y, w, h]})
                if save_visualize and frame_vis_dir: frame_log_data['status'] = 'success'
            else:
                # 预测或转换失败，不更新 current_frame 和 current_bbox_xyxy
                # 写入上一帧的结果，以保持文件行数一致性
                with open(pred_file_path, "a") as f:
                    prev_x, prev_y = current_bbox_xyxy[0], current_bbox_xyxy[1]
                    prev_w, prev_h = current_bbox_xyxy[2]-prev_x, current_bbox_xyxy[3]-prev_y
                    f.write(f"{prev_x},{prev_y},{prev_w},{prev_h}\n")
                seq_results.append({'frame_id': i, 'bbox': [prev_x, prev_y, prev_w, prev_h]}) # 标记失败，但写入旧bbox
                # 状态已在上面设置


            # 保存日志
            if save_visualize and frame_vis_dir:
                log_file_path = os.path.join(frame_vis_dir, "log_data.json")
                with open(log_file_path, "w") as f_log:
                    def default_serializer(obj):
                        if isinstance(obj, (np.ndarray, Image.Image)): return str(obj)
                        try: return json.JSONEncoder.default(self, obj)
                        except TypeError: return str(obj)
                    json.dump(frame_log_data, f_log, indent=2, default=default_serializer)

            process_times["total_frame"].append(time.time() - frame_start_time)

        results.append({'sequence_name': seq.name, 'frames': seq_results})

    # 打印性能统计
    logger.info("--- Performance Stats (No Crop) ---")
    for key, times in process_times.items():
        if times: logger.info(f"Average {key} time: {sum(times) / len(times):.4f}s")
    total_avg_time = sum(process_times["total_frame"]) / len(process_times["total_frame"]) if process_times["total_frame"] else 0
    fps = 1.0 / total_avg_time if total_avg_time > 0 else 0
    logger.info(f"Average FPS: {fps:.2f}")
    logger.info("------------------------------------")

    results_json_path = os.path.join(output_dir, f"{dataset_name}_results_no_crop.json")
    with open(results_json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"No-crop results saved to {results_json_path}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Test Tracking Model without Cropping")
    parser.add_argument("--model_path", type=str,
                        default='/data1/lihaobo/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/tracking_large-crop-2025-4-25',
                        help="Path to the fully fine-tuned model directory")
    parser.add_argument("--dataset_name", type=str, default="OTB_lang", help="Dataset name for evaluation")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--sequence", type=str, default='Biker', help="Specific sequence to test (optional)")
    parser.add_argument("--save_vis", action="store_true", default=True, help="Save visualization results")
    # 移除裁剪相关参数
    # parser.add_argument("--template_scale", type=float, default=2.0, help="Scale factor for template cropping")
    # parser.add_argument("--search_scale", type=float, default=4.0, help="Scale factor for search region cropping")
    # parser.add_argument("--resize", type=int, default=320, help="Size to resize cropped images")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate")

    args = parser.parse_args()

    if args.output_dir is None:
        model_name = os.path.basename(args.model_path.rstrip('/'))
        args.output_dir = f"results_{args.dataset_name}_{model_name}_no_crop" # 更新目录名
    os.makedirs(args.output_dir, exist_ok=True)

    model, tokenizer, processor = load_model(args.model_path)
    sequences = [args.sequence] if args.sequence else None

    # 调用修改后的评估函数
    evaluate_tracking(
        model, tokenizer, processor,
        dataset_name=args.dataset_name, sequences=sequences,
        save_visualize=args.save_vis,
        output_dir=args.output_dir, max_new_tokens=args.max_new_tokens
    )

    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()