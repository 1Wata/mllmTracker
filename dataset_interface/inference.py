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
from qwen_vl_utils import process_vision_info
from peft import PeftModel

from make_crop_dataset.utils import convert_bbox_from_cropped_img
from make_crop_dataset.utils import crop_and_pad_template, crop_and_pad_search, convert_bbox_format
from evaluation.datasets import get_dataset, SequenceList

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(base_model_path, adapter_path, device="cuda"):
    """加载基础模型和LoRA适配器"""
    logger.info(f"Loading base model from {base_model_path}")
    # 加载基础模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device # 使用 device_map 加载基础模型到指定设备
    )
    
    logger.info(f"Loading LoRA adapter from {adapter_path}")
    # 加载LoRA适配器
    # 注意：确保 PeftModel 加载时基础模型已经在目标设备上
    model = PeftModel.from_pretrained(model, adapter_path, device_map=device) 
    
    # 合并LoRA权重 (可选，但推荐用于推理以提高速度)
    # 如果不合并，模型会动态应用LoRA层，可能会稍慢
    logger.info("Merging LoRA adapter...")
    model = model.merge_and_unload()
    logger.info("LoRA adapter merged.")

    # 从基础模型路径加载分词器和处理器
    logger.info(f"Loading tokenizer and processor from {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    processor = AutoProcessor.from_pretrained(base_model_path)

    model.eval()
    return model, tokenizer, processor

def extract_bbox_from_response(response):
    """从模型响应中提取边界框坐标"""
    patterns = [
        r'\[(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+)\]',
        r'\[\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*\]'
    ]
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, response))
        if matches:
            # 获取最后一个匹配
            last_match = matches[-1]
            
            # 将坐标转换为浮点数并保留一位小数
            x1 = round(float(last_match.group(1)), 1)
            y1 = round(float(last_match.group(2)), 1)
            x2 = round(float(last_match.group(3)), 1)
            y2 = round(float(last_match.group(4)), 1)
            
            return [x1, y1, x2, y2]
    
    logger.warning(f"Failed to extract bounding box from response: {response}")
    return None

def draw_bbox(image, bbox, color="red", width=2):
    """在图像上绘制边界框"""
    draw = ImageDraw.Draw(image)
    draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=color, width=width)
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
    
    # 返回完整对话
    full_response = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return full_response[0]

def build_cropped_test_message(template_imgs, search_img, template_bboxes, exp_str):
    """
    构建测试消息 - 使用裁剪后的图像
    
    Args:
        template_imgs: 模板图像列表 (已裁剪)
        search_img: 搜索图像 (已裁剪)
        template_bboxes: 模板图像中的边界框列表
        exp_str: 目标描述
        
    Returns:
        构建好的消息列表
    """
    messages = []
    
    # 如果有多个模板帧
    if len(template_imgs) > 1:
        user_content = (f"The first {len(template_imgs)} images show the object of interest: '{exp_str}'. "
                        f"Please locate this object in the final image. "
                        f"Provide its bounding box as [x1, y1, x2, y2] coordinates within that image.")
    else:
        user_content = (f"This image shows the object of interest: '{exp_str}'. "
                        f"Please locate this object in the following image. "
                        f"Provide its bounding box as [x1, y1, x2, y2] coordinates within that image.")
    
    messages.append({"role": "user", "content": user_content})
    return messages

def evaluate_tracking_cropped(model, tokenizer, processor, dataset_name="lasot", 
                             sequences=None, template_crop_scale=2.0, search_crop_scale=4.0, 
                             resize=320, save_visualize=False, output_dir=None, max_new_tokens=2048):
    """
    使用裁剪图像策略评估跟踪性能 - 采用第一帧和最近的一个可见帧作为模板
    
    Args:
        model: 模型
        tokenizer: 分词器
        processor: 处理器
        dataset_name: 数据集名称
        sequences: 要处理的特定序列名称列表
        template_crop_scale: 模板图像裁剪的缩放比例
        search_crop_scale: 搜索图像裁剪的缩放比例
        resize: 裁剪后图像的尺寸
        save_visualize: 是否保存可视化结果
        output_dir: 输出目录
        max_new_tokens: 生成的最大新token数
        
    Returns:
        跟踪结果
    """
    if output_dir is None:
        output_dir = f"tracking_results/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    vis_dir = None
    if save_visualize:
        vis_dir = os.path.join(output_dir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
    
    # 加载数据集
    dataset = get_dataset(dataset_name)
    results = []
    
    # 过滤序列（如果指定）
    if sequences:
        filtered_dataset = []
        for seq_name in sequences:
            for seq in dataset:
                if seq.name == seq_name:
                    filtered_dataset.append(seq)
                    break
        
        if filtered_dataset:
            dataset = SequenceList(filtered_dataset)
        else:
            logger.warning(f"None of the specified sequences found in dataset {dataset_name}") # Added dataset_name for clarity
            return []
    
    logger.info(f"Processing {len(dataset)} sequences from {dataset_name}") # Added dataset_name
    
    # 性能计时记录
    process_times = {
        "image_loading": [],
        "template_crop": [],
        "search_crop": [],
        "model_inference": [],
        "total_frame": []
    }
    
    for seq in tqdm(dataset, desc="Tracking progress"):
        seq_results = []
        
        # 获取第一帧及其边界框
        init_info = seq.init_info()
        first_frame_path = seq.frames[0]
        first_frame_bbox = init_info.get('init_bbox')
        exp_str = init_info.get('init_text_description', 'the target object') # Added default description
        
        # 创建预测输出文件 (for evaluation scripts)
        pred_file_path = os.path.join(output_dir, f"{seq.name}.txt") # Renamed from gt_file_path
        with open(pred_file_path, "w") as f:
            x, y, w, h = first_frame_bbox
            f.write(f"{x},{y},{w},{h}\n")
        
        # 转换bbox格式 [x, y, w, h] -> [x1, y1, x2, y2]
        first_frame_bbox_xyxy = [
            first_frame_bbox[0], 
            first_frame_bbox[1], 
            first_frame_bbox[0] + first_frame_bbox[2], 
            first_frame_bbox[1] + first_frame_bbox[3]
        ]
        
        # 创建序列可视化目录
        seq_vis_dir = None
        if save_visualize and vis_dir: # Check vis_dir exists
            seq_vis_dir = os.path.join(vis_dir, seq.name)
            os.makedirs(seq_vis_dir, exist_ok=True)
        
        # 加载第一帧
        try:
            first_frame = Image.open(first_frame_path).convert("RGB") # Ensure RGB
        except FileNotFoundError:
            logger.error(f"First frame not found: {first_frame_path}. Skipping sequence {seq.name}.")
            continue
        except Exception as e:
            logger.error(f"Error loading first frame {first_frame_path}: {e}. Skipping sequence {seq.name}.")
            continue

        # 裁剪第一帧
        try:
            cropped_first_frame = crop_and_pad_template(
                first_frame, first_frame_bbox_xyxy, 
                scale=template_crop_scale, resize=resize
            )
        except Exception as e:
             logger.error(f"Error cropping first frame template for sequence {seq.name}: {e}. Skipping sequence.")
             continue

        # 初始化追踪状态
        current_bbox_xyxy = first_frame_bbox_xyxy
        current_frame = first_frame
        
        # 保存第一帧可视化
        if save_visualize and seq_vis_dir:
            frame_0_dir = os.path.join(seq_vis_dir, "frame_0000")
            os.makedirs(frame_0_dir, exist_ok=True)
            try:
                first_frame_vis = draw_bbox(first_frame.copy(), first_frame_bbox_xyxy, color="green") # Use green for GT
                first_frame_vis.save(os.path.join(frame_0_dir, f"search_original_with_gt_bbox.jpg"))
                cropped_first_frame.save(os.path.join(frame_0_dir, f"template_0000.jpg"))
            except Exception as e:
                 logger.warning(f"Could not save visualization for frame 0 of {seq.name}: {e}")

        # 循环处理后续帧
        for i, frame_path in enumerate(seq.frames[1:], start=1):
            frame_start_time = time.time()
            frame_log_data = {} # 用于存储文本日志信息
            frame_vis_dir = None
            if save_visualize and seq_vis_dir:
                frame_vis_dir = os.path.join(seq_vis_dir, f"frame_{i:04d}")
                os.makedirs(frame_vis_dir, exist_ok=True)

            # 加载当前帧
            t_start = time.time()
            try:
                search_frame = Image.open(frame_path).convert("RGB") # Ensure RGB
                process_times["image_loading"].append(time.time() - t_start)
                if save_visualize and frame_vis_dir:
                    search_frame.save(os.path.join(frame_vis_dir, "search_original.jpg"))
            except FileNotFoundError:
                logger.error(f"Frame {i} not found: {frame_path}. Writing previous bbox and skipping frame.")
                # Write previous bbox and continue
                with open(pred_file_path, "a") as f:
                    x, y = current_bbox_xyxy[0], current_bbox_xyxy[1]
                    w, h = current_bbox_xyxy[2]-current_bbox_xyxy[0], current_bbox_xyxy[3]-current_bbox_xyxy[1]
                    f.write(f"{x},{y},{w},{h}\n")
                seq_results.append({'frame_id': i, 'bbox': [x, y, w, h], 'status': 'load_failed'})
                continue
            except Exception as e:
                logger.error(f"Error loading frame {i} ({frame_path}): {e}. Writing previous bbox and skipping frame.")
                # Write previous bbox and continue
                with open(pred_file_path, "a") as f:
                    x, y = current_bbox_xyxy[0], current_bbox_xyxy[1]
                    w, h = current_bbox_xyxy[2]-current_bbox_xyxy[0], current_bbox_xyxy[3]-current_bbox_xyxy[1]
                    f.write(f"{x},{y},{w},{h}\n")
                seq_results.append({'frame_id': i, 'bbox': [x, y, w, h], 'status': 'load_failed'})
                continue


            # 准备模板帧
            t_start_template = time.time()
            if i == 1:
                cropped_templates = [cropped_first_frame, cropped_first_frame]
            else:
                try:
                    cropped_current_template = crop_and_pad_template(
                        current_frame, current_bbox_xyxy,
                        scale=template_crop_scale, resize=resize
                    )
                    cropped_templates = [cropped_first_frame, cropped_current_template]
                except Exception as e: # Catch potential errors during template cropping
                     logger.warning(f"Error cropping template frame {i-1} for current frame {i} in sequence {seq.name}: {e}. Using first frame template twice.")
                     cropped_templates = [cropped_first_frame, cropped_first_frame] # Fallback
            process_times["template_crop"].append(time.time() - t_start_template)

            # 保存裁剪的模板帧
            if save_visualize and frame_vis_dir:
                try:
                    cropped_templates[0].save(os.path.join(frame_vis_dir, "template_first.jpg"))
                    cropped_templates[1].save(os.path.join(frame_vis_dir, "template_last.jpg"))
                except Exception as e:
                    logger.warning(f"Could not save template visualization for frame {i} of {seq.name}: {e}")

            # 裁剪搜索帧
            t_start_search = time.time()
            crop_region = None # Initialize crop_region
            cropped_search_img = None
            try:
                cropped_search_img, _, crop_region, _ = crop_and_pad_search(
                    search_frame, current_bbox_xyxy, None,
                    scale=search_crop_scale, resize=resize
                )
                process_times["search_crop"].append(time.time() - t_start_search)
                if save_visualize and frame_vis_dir:
                    cropped_search_img.save(os.path.join(frame_vis_dir, "search_cropped.jpg"))

            except Exception as e: # Catch potential errors during search cropping
                logger.warning(f"Error cropping search frame {i} in sequence {seq.name}: {e}. Writing previous bbox and skipping inference.")
                # 预测失败，写入上一帧的结果
                with open(pred_file_path, "a") as f:
                    x, y = current_bbox_xyxy[0], current_bbox_xyxy[1]
                    w, h = current_bbox_xyxy[2]-current_bbox_xyxy[0], current_bbox_xyxy[3]-current_bbox_xyxy[1]
                    f.write(f"{x},{y},{w},{h}\n")
                
                seq_results.append({
                    'frame_id': i,
                    'bbox': [x, y, w, h],
                    'status': 'crop_failed'
                })
                # 保存日志
                if save_visualize and frame_vis_dir:
                    frame_log_data['status'] = 'crop_failed'
                    frame_log_data['error'] = str(e)
                    log_file_path = os.path.join(frame_vis_dir, "log_data.json")
                    try:
                        with open(log_file_path, "w") as f_log:
                            json.dump(frame_log_data, f_log, indent=2)
                    except Exception as log_e:
                         logger.warning(f"Could not write log file for frame {i} of {seq.name}: {log_e}")
                continue # 跳过模型推理

            # 构建消息
            messages = build_cropped_test_message(
                cropped_templates, cropped_search_img, 
                [None, None], # Template bboxes not needed in message for this strategy
                exp_str
            )
            
            # 处理输入
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Combine images for processor
            model_input_images = cropped_templates + [cropped_search_img]
            image_inputs, _, _ = process_vision_info([messages], model_input_images) # Pass combined images

            try:
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            except Exception as e:
                logger.error(f"Error processing inputs for frame {i}, seq {seq.name}: {e}. Writing previous bbox.")
                with open(pred_file_path, "a") as f:
                    x, y = current_bbox_xyxy[0], current_bbox_xyxy[1]
                    w, h = current_bbox_xyxy[2]-current_bbox_xyxy[0], current_bbox_xyxy[3]-current_bbox_xyxy[1]
                    f.write(f"{x},{y},{w},{h}\n")
                seq_results.append({'frame_id': i, 'bbox': [x, y, w, h], 'status': 'processor_failed'})
                if save_visualize and frame_vis_dir: frame_log_data['status'] = 'processor_failed'; frame_log_data['error'] = str(e)
                # Save log before continuing
                if save_visualize and frame_vis_dir:
                    log_file_path = os.path.join(frame_vis_dir, "log_data.json")
                    try:
                        with open(log_file_path, "w") as f_log: json.dump(frame_log_data, f_log, indent=2)
                    except Exception as log_e: logger.warning(f"Could not write log file for frame {i} of {seq.name}: {log_e}")
                continue


            # 保存输入提示
            if save_visualize and frame_vis_dir:
                frame_log_data['input_prompt'] = text
            
            # 生成输出
            t_start_inference = time.time()
            response = None
            try:
                response = generate_output(model, tokenizer, inputs, max_new_tokens)
                process_times["model_inference"].append(time.time() - t_start_inference)
            except Exception as e:
                 logger.error(f"Error during model inference for frame {i}, seq {seq.name}: {e}. Writing previous bbox.")
                 with open(pred_file_path, "a") as f:
                    x, y = current_bbox_xyxy[0], current_bbox_xyxy[1]
                    w, h = current_bbox_xyxy[2]-current_bbox_xyxy[0], current_bbox_xyxy[3]-current_bbox_xyxy[1]
                    f.write(f"{x},{y},{w},{h}\n")
                 seq_results.append({'frame_id': i, 'bbox': [x, y, w, h], 'status': 'inference_failed'})
                 if save_visualize and frame_vis_dir: frame_log_data['status'] = 'inference_failed'; frame_log_data['error'] = str(e)
                 # Save log before continuing
                 if save_visualize and frame_vis_dir:
                    log_file_path = os.path.join(frame_vis_dir, "log_data.json")
                    try:
                        with open(log_file_path, "w") as f_log: json.dump(frame_log_data, f_log, indent=2)
                    except Exception as log_e: logger.warning(f"Could not write log file for frame {i} of {seq.name}: {log_e}")
                 continue


            # 保存模型原始输出
            if save_visualize and frame_vis_dir:
                frame_log_data['model_response'] = response

            # 提取边界框
            predicted_bbox = extract_bbox_from_response(response)
            abs_bbox = None # Initialize absolute bbox

            if predicted_bbox:
                 if save_visualize and frame_vis_dir:
                    frame_log_data['predicted_bbox_cropped'] = predicted_bbox
                    # --- VISUALIZATION 1: Draw predicted_bbox on cropped_search_img ---
                    try:
                        vis_cropped = draw_bbox(cropped_search_img.copy(), predicted_bbox, color="blue") # Blue for prediction
                        vis_cropped.save(os.path.join(frame_vis_dir, "search_cropped_with_pred_bbox.jpg"))
                    except Exception as draw_err:
                         logger.warning(f"Error drawing bbox on cropped image for frame {i}, seq {seq.name}: {draw_err}")
                         frame_log_data['vis_crop_error'] = str(draw_err)
                 # --- End Visualization 1 ---

                 # 将预测的边界框转换回原始图像坐标
                 if crop_region: # Ensure cropping was successful
                    try:
                        abs_bbox = convert_bbox_from_cropped_img(crop_region, predicted_bbox, resize)
                        
                        if save_visualize and frame_vis_dir:
                            frame_log_data['predicted_bbox_original'] = abs_bbox

                        # 更新当前帧和边界框 (用于下一帧的裁剪)
                        current_frame = search_frame
                        current_bbox_xyxy = abs_bbox
                        
                        # 转换为 [x, y, w, h] 格式用于保存结果
                        x, y, w, h = abs_bbox[0], abs_bbox[1], abs_bbox[2]-abs_bbox[0], abs_bbox[3]-abs_bbox[1]
                        
                        # 保存到预测文件
                        with open(pred_file_path, "a") as f:
                            f.write(f"{x},{y},{w},{h}\n")
                        
                        seq_results.append({
                            'frame_id': i,
                            'bbox': [x, y, w, h],
                            'status': 'success'
                        })
                        frame_log_data['status'] = 'success'
                        
                        # --- VISUALIZATION 2: Draw abs_bbox on search_frame ---
                        if save_visualize and frame_vis_dir:
                            try:
                                search_vis = draw_bbox(search_frame.copy(), abs_bbox, color="red") # Red for final prediction
                                search_vis.save(os.path.join(frame_vis_dir, "search_original_with_pred_bbox.jpg"))
                            except Exception as draw_err:
                                logger.warning(f"Error drawing bbox on original image for frame {i}, seq {seq.name}: {draw_err}")
                                frame_log_data['vis_original_error'] = str(draw_err)
                        # --- End Visualization 2 ---

                    except Exception as convert_err:
                        logger.error(f"Error converting bbox for frame {i}, seq {seq.name}: {convert_err}. Writing previous bbox.")
                        # 转换失败，视为预测失败，写入上一帧结果
                        with open(pred_file_path, "a") as f:
                            x, y = current_bbox_xyxy[0], current_bbox_xyxy[1]
                            w, h = current_bbox_xyxy[2]-current_bbox_xyxy[0], current_bbox_xyxy[3]-current_bbox_xyxy[1]
                            f.write(f"{x},{y},{w},{h}\n")
                        seq_results.append({
                            'frame_id': i,
                            'bbox': [x, y, w, h],
                            'status': 'conversion_failed'
                        })
                        if save_visualize and frame_vis_dir:
                            frame_log_data['status'] = 'conversion_failed'
                            frame_log_data['error'] = str(convert_err)
                 else:
                    # Should not happen if predicted_bbox is not None, but as a safeguard
                    logger.warning(f"Predicted bbox found but no crop_region for frame {i}, seq {seq.name}. Writing previous bbox.")
                    with open(pred_file_path, "a") as f:
                        x, y = current_bbox_xyxy[0], current_bbox_xyxy[1]
                        w, h = current_bbox_xyxy[2]-current_bbox_xyxy[0], current_bbox_xyxy[3]-current_bbox_xyxy[1]
                        f.write(f"{x},{y},{w},{h}\n")
                    seq_results.append({'frame_id': i, 'bbox': [x, y, w, h], 'status': 'no_crop_region_post_pred'})
                    if save_visualize and frame_vis_dir: frame_log_data['status'] = 'no_crop_region_post_pred'

            else:
                # 提取边界框失败
                logger.warning(f"Failed to extract bbox from response for frame {i} in sequence {seq.name}. Writing previous bbox.")
                # 写入上一帧的结果
                with open(pred_file_path, "a") as f:
                    x, y = current_bbox_xyxy[0], current_bbox_xyxy[1]
                    w, h = current_bbox_xyxy[2]-current_bbox_xyxy[0], current_bbox_xyxy[3]-current_bbox_xyxy[1]
                    f.write(f"{x},{y},{w},{h}\n")
                
                seq_results.append({
                    'frame_id': i,
                    'bbox': [x, y, w, h],
                    'status': 'extraction_failed'
                })
                if save_visualize and frame_vis_dir:
                    frame_log_data['status'] = 'extraction_failed'

            # 保存当前帧的所有文本日志信息
            if save_visualize and frame_vis_dir:
                 log_file_path = os.path.join(frame_vis_dir, "log_data.json")
                 try:
                     with open(log_file_path, "w") as f_log:
                         # Convert numpy arrays/PIL images if any accidentally got in log_data
                         def default_serializer(obj):
                             if isinstance(obj, (np.ndarray, Image.Image)):
                                 return str(obj) # Or a more meaningful representation
                             raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
                         json.dump(frame_log_data, f_log, indent=2, default=default_serializer)
                 except Exception as log_e:
                      logger.warning(f"Could not write log file for frame {i} of {seq.name}: {log_e}")


            process_times["total_frame"].append(time.time() - frame_start_time)
            
        results.append({
            'sequence_name': seq.name,
            'frames': seq_results
        })
    
    # 打印性能统计
    logger.info("--- Performance Stats ---")
    for key, times in process_times.items():
        if times:
            avg_time = sum(times) / len(times)
            logger.info(f"Average {key} time: {avg_time:.4f}s")
        else:
            logger.info(f"No data for {key} time.")
    logger.info("-------------------------")

    # 保存整体结果为JSON (可选)
    results_json_path = os.path.join(output_dir, f"{dataset_name}_results.json")
    try:
        with open(results_json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Overall results saved to {results_json_path}")
    except Exception as e:
        logger.error(f"Could not save overall results JSON: {e}")

    return results

def main():
    parser = argparse.ArgumentParser(description="Test Tracking Model with Cropped Images Strategy")
    # 重命名 model_path 为 adapter_path
    parser.add_argument("--adapter_path", type=str, default='/data1/lihaobo/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/lora/tracking_large-crop-2025-4-25',
                        help="Path to the LoRA adapter weights")
    # 添加基础模型路径参数
    parser.add_argument("--base_model_path", type=str, default='/data1/lihaobo/LLaMA-Factory/pretrained/Qwen2.5-VL-3B-Instruct',
                        help="Path to the base model (e.g., Qwen/Qwen2.5-VL-3B-Instruct)")
    parser.add_argument("--dataset_name", type=str, default="OTB_lang",
                        help="Dataset name for evaluation")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--sequence", type=str, default='Biker',
                        help="Specific sequence to test (optional)")
    parser.add_argument("--save_vis", action="store_true", default=True,
                        help="Save visualization results")
    parser.add_argument("--template_scale", type=float, default=2.0,
                        help="Scale factor for template cropping")
    parser.add_argument("--search_scale", type=float, default=4.0,
                        help="Scale factor for search region cropping")
    parser.add_argument("--resize", type=int, default=320,
                        help="Size to resize cropped images")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum new tokens to generate")
    
    args = parser.parse_args()
    
    # 设置默认输出目录
    if args.output_dir is None:
        # 可以基于适配器名称生成输出目录
        adapter_name = os.path.basename(args.adapter_path)
        args.output_dir = f"results_{args.dataset_name}_{adapter_name}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载模型，传入基础模型路径和适配器路径
    model, tokenizer, processor = load_model(args.base_model_path, args.adapter_path)
    
    # 设置要处理的序列
    sequences = [args.sequence] if args.sequence else None
    
    # 运行评估
    results = evaluate_tracking_cropped(
        model,
        tokenizer,
        processor,
        dataset_name=args.dataset_name,
        sequences=sequences,
        template_crop_scale=args.template_scale,
        search_crop_scale=args.search_scale,
        resize=args.resize,
        save_visualize=args.save_vis,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens
    )
    
    logger.info(f"Evaluation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    # Add basic error handling for main execution
    try:
        main()
    except Exception as e:
        logger.exception(f"An error occurred during script execution: {e}") # Log traceback
        exit(1) # Exit with error code