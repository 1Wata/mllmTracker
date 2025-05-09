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
from transformers import AutoProcessor, GenerationConfig
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rft/src/virft/src'))
)

from open_r1.utils.utils import transform_bbox, smart_resize
from evaluation.datasets import get_dataset, SequenceList

import functools
import multiprocessing as mp
from multiprocessing import Pool
import math
from typing import List, Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_FACTOR = 28
DEFAULT_MIN_PIXELS = 3136
DEFAULT_MAX_PIXELS = 102400

# --- Add TRACKING_SYSTEM_PROMPT ---
def get_tracking_system_prompt(use_thinking=False):
    """获取追踪任务的系统提示"""
    base_prompt = (
        "You are a professional visual object tracking assistant. Your task is to track a specified target object. "
        "The user will provide template frames showing the target object. "
        "First, identify the target in the template frames. Then, locate the target's position in the following search frames."
    )
    
    if use_thinking: 
        return base_prompt + (
            "You should first think about how to locate the target by analyzing visual features, then provide your answer. "
            "Put your thinking process inside <thinking>...</thinking> tags. "
            "Your final answer should be the target's bounding box coordinates in the format [x1, y1, x2, y2], "
            "where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate. "
            "Wrap your final answer in <answer>[x1, y1, x2, y2]</answer> tags."
        )
    else:
        return base_prompt + (
            "Please directly return the target's bounding box coordinates in the format [x1, y1, x2, y2], "
            "where (x1, y1) is the top-left coordinate and (x2, y2) is the bottom-right coordinate. "
            "Your answer should be wrapped in <answer>[x1, y1, x2, y2]</answer> tags."
        )

TRACKING_SYSTEM_PROMPT = get_tracking_system_prompt(False)
# --- End ---

def convert_bbox_xywh_to_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return [x, y, x + w, y + h]

def convert_bbox_xyxy_to_xywh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy
    return [x1, y1, x2 - x1, y2 - y1]

def extract_single_bbox(response_text: str) -> Optional[List[int]]:
    answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)
    content = answer_match.group(1).strip() if answer_match else response_text.strip()
    bbox_match = re.search(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", content)
    if bbox_match:
        return [int(coord) for coord in bbox_match.groups()]
    logger.warning(f"Simplified: Failed to extract bounding box from response: {response_text}")
    return None

def load_model_and_processor(model_path, device="auto"):
    logger.info(f"Loading model from {model_path} to device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
        attn_implementation="flash_attention_2"
    )
    model.eval()
    return model, tokenizer, processor

def build_rft_input_messages(exp_str, template_pils_resized, search_pil_resized, template_bboxes_for_text_prompt):
    user_content_list_of_dicts = []
    for _ in template_pils_resized:
        user_content_list_of_dicts.append({"type": "image"})
    if template_pils_resized:
        for idx, template_bbox_in_resized_coords in enumerate(template_bboxes_for_text_prompt):
            bbox_text_content = (
                f"The bounding box for template frame {idx + 1} is: "
                f"[{int(template_bbox_in_resized_coords[0])}, {int(template_bbox_in_resized_coords[1])}, "
                f"{int(template_bbox_in_resized_coords[2])}, {int(template_bbox_in_resized_coords[3])}]."
            )
            user_content_list_of_dicts.append({"type": "text", "text": "\n" + bbox_text_content})
    if template_pils_resized:
        object_description_text = f"\nThese are the template frames showing the object '{exp_str}'."
        user_content_list_of_dicts.append({"type": "text", "text": object_description_text})
    user_content_list_of_dicts.append({"type": "image"})
    tracking_instruction_text = (
        f" Please track the object '{exp_str}' in the search frame provided after the template frames. "
        "Provide a bounding box for this search frame. "
        "The bounding box should be in [x1, y1, x2, y2] format, relative to this search frame's resized dimensions. "
        "Wrap your answer in <answer>[x1, y1, x2, y2]</answer> tags."
    )
    user_content_list_of_dicts.append({"type": "text", "text": tracking_instruction_text})
    messages = [
        {"role": "system", "content": TRACKING_SYSTEM_PROMPT},
        {"role": "user", "content": user_content_list_of_dicts}
    ]
    return messages

def draw_bbox_on_image(image_pil, bbox_xyxy, color="red", width=2):
    if bbox_xyxy is None:
        return image_pil
    img_draw = image_pil.copy()
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle(bbox_xyxy, outline=color, width=width)
    return img_draw

def update_history_indexed(
    history_frame_paths: List[Optional[str]], 
    history_bboxes_orig_xyxy: List[Optional[List[int]]], 
    current_frame_idx: int,
    current_frame_path: Optional[str], 
    current_bbox_orig_xyxy: Optional[List[int]]
):
    """Updates history with frame path and bbox at a specific frame index."""
    while len(history_frame_paths) <= current_frame_idx:
        history_frame_paths.append(None)
    while len(history_bboxes_orig_xyxy) <= current_frame_idx:
        history_bboxes_orig_xyxy.append(None)
    
    history_frame_paths[current_frame_idx] = current_frame_path
    history_bboxes_orig_xyxy[current_frame_idx] = current_bbox_orig_xyxy
    return history_frame_paths, history_bboxes_orig_xyxy

def get_template_frames_by_gap(
    current_frame_idx: int,
    gap_list: List[int],
    all_history_frame_paths: List[Optional[str]], 
    all_history_bboxes_orig_xyxy: List[Optional[List[int]]], 
    first_frame_path: str, 
    first_frame_bbox_orig_xyxy: List[int] 
) -> Tuple[List[Image.Image], List[List[int]]]:
    selected_template_pils_orig = []
    selected_template_bboxes_orig_xyxy = []
    temp_selected_paths_bboxes = []

    for gap in gap_list:
        if gap <= 0: 
            logger.warning(f"Invalid gap value {gap} <= 0, skipping.")
            continue
        template_frame_idx = current_frame_idx - gap
        path_to_load = None
        bbox_to_use = None

        if template_frame_idx >= 0:
            if template_frame_idx < len(all_history_frame_paths) and all_history_frame_paths[template_frame_idx] is not None:
                path_to_load = all_history_frame_paths[template_frame_idx]
                bbox_to_use = all_history_bboxes_orig_xyxy[template_frame_idx]
            else:
                logger.warning(f"History missing for frame index {template_frame_idx} (current_frame_idx: {current_frame_idx}, gap: {gap}). Using first frame path as fallback.")
                path_to_load = first_frame_path
                bbox_to_use = first_frame_bbox_orig_xyxy
        else: 
            logger.info(f"History not long enough for gap {gap} at current frame {current_frame_idx}. Using first frame path.")
            path_to_load = first_frame_path
            bbox_to_use = first_frame_bbox_orig_xyxy
        
        if path_to_load and bbox_to_use:
            is_duplicate_path = any(p == path_to_load for p, _ in temp_selected_paths_bboxes)
            if not is_duplicate_path:
                temp_selected_paths_bboxes.append((path_to_load, bbox_to_use))
            else:
                logger.info(f"Skipping duplicate template path '{path_to_load}' for gap {gap}.")
    
    for path, bbox in temp_selected_paths_bboxes:

        pil_image = Image.open(path).convert("RGB")
        selected_template_pils_orig.append(pil_image)
        selected_template_bboxes_orig_xyxy.append(bbox)

    if not selected_template_pils_orig and current_frame_idx > 0 :
        logger.info("No templates selected via gap_list or loading failed, defaulting to first frame path as the sole template.")

        pil_image = Image.open(first_frame_path).convert("RGB")
        selected_template_pils_orig.append(pil_image)
        selected_template_bboxes_orig_xyxy.append(first_frame_bbox_orig_xyxy)
            
    return selected_template_pils_orig, selected_template_bboxes_orig_xyxy

def evaluate_tracking_rft(model, tokenizer, processor, sequences_names=None, dataset_name='lasot',
                          save_visualize=False, output_dir="rft_tracking_results", 
                          max_new_tokens=100, rank=0, 
                          smart_resize_factor=DEFAULT_FACTOR, 
                          smart_resize_min_pixels=DEFAULT_MIN_PIXELS, 
                          smart_resize_max_pixels=DEFAULT_MAX_PIXELS,
                          gap_list: Optional[List[int]] = None):
    
    if gap_list is None:
        gap_list = [1, 5] 
        logger.info(f"gap_list not provided, using default: {gap_list}")

    dataset = get_dataset(dataset_name)
    results_summary = []

    if sequences_names:
        filtered_dataset_seqs = [dataset[seq_name] for seq_name in sequences_names if seq_name in dataset.seq_names]
        if not filtered_dataset_seqs:
            logger.warning(f"Process {rank} - No valid sequences found to process from provided list.")
            return []
        dataset = SequenceList(filtered_dataset_seqs)
    
    logger.info(f"Process {rank} - Processing {len(dataset)} sequences (simplified) with gap_list: {gap_list}.")

    for seq_idx, seq in enumerate(tqdm(dataset, desc=f"Process {rank} - Tracking progress")):
        seq_output_dir = os.path.join(output_dir, dataset_name, seq.name)
        os.makedirs(seq_output_dir, exist_ok=True)
        predictions_file_path = os.path.join(seq_output_dir, "predictions.txt")
        
        all_history_frame_paths: List[Optional[str]] = [] 
        all_history_bboxes_orig_xyxy: List[Optional[List[int]]] = []

        first_frame_path = seq.frames[0]
        first_frame_pil = Image.open(first_frame_path).convert("RGB")
        init_info = seq.init_info()
        first_frame_gt_bbox_xywh = init_info.get('init_bbox')
        first_frame_gt_bbox_original_xyxy = convert_bbox_xywh_to_xyxy(first_frame_gt_bbox_xywh)
        exp_str = init_info.get('init_text_description', "the target object")

        original_size_first = first_frame_pil.size
        resized_h_first, resized_w_first = smart_resize(
            original_size_first[1], original_size_first[0], 
            smart_resize_factor, smart_resize_min_pixels, smart_resize_max_pixels
        )
        resized_size_first = (resized_w_first, resized_h_first)
        first_frame_bbox_for_text_prompt = transform_bbox(
            first_frame_gt_bbox_original_xyxy, original_size_first, resized_size_first, 'original_to_resized'
        )

        if save_visualize:
            img_draw = draw_bbox_on_image(first_frame_pil, first_frame_gt_bbox_original_xyxy)
            img_draw.save(os.path.join(seq_output_dir, f"frame_0000_gt.jpg"))

        with open(predictions_file_path, "w") as f:
            f.write(f"{first_frame_gt_bbox_xywh[0]},{first_frame_gt_bbox_xywh[1]},{first_frame_gt_bbox_xywh[2]},{first_frame_gt_bbox_xywh[3]}\n")
        
        current_pred_bbox_original_xyxy_for_history = list(first_frame_gt_bbox_original_xyxy)
        all_history_frame_paths, all_history_bboxes_orig_xyxy = update_history_indexed(
            all_history_frame_paths, all_history_bboxes_orig_xyxy,
            0, first_frame_path, current_pred_bbox_original_xyxy_for_history
        )

        for i in range(1, len(seq.frames)):
            current_prompt_template_pils_resized = [] 
            current_prompt_template_bboxes_for_text = []
            
            selected_template_pils_orig, selected_template_bboxes_orig_xyxy = get_template_frames_by_gap(
                current_frame_idx=i,
                gap_list=gap_list,
                all_history_frame_paths=all_history_frame_paths,
                all_history_bboxes_orig_xyxy=all_history_bboxes_orig_xyxy,
                first_frame_path=first_frame_path, 
                first_frame_bbox_orig_xyxy=first_frame_gt_bbox_original_xyxy
            )
            
            if not selected_template_pils_orig:
                 logger.warning(f"No template frames could be prepared for frame {i}. Using first frame as sole prompt template.")
                 resized_first_pil_for_prompt = first_frame_pil.resize(resized_size_first, Image.Resampling.BILINEAR)
                 current_prompt_template_pils_resized = [resized_first_pil_for_prompt]
                 current_prompt_template_bboxes_for_text = [first_frame_bbox_for_text_prompt]
            else:
                for template_pil_orig, template_bbox_orig_xyxy in zip(selected_template_pils_orig, selected_template_bboxes_orig_xyxy):
                    if template_pil_orig is None or template_bbox_orig_xyxy is None: 
                        logger.warning(f"Encountered None in selected templates for frame {i}. Skipping this template.")
                        continue
                    orig_s = template_pil_orig.size
                    res_h, res_w = smart_resize(orig_s[1], orig_s[0], smart_resize_factor, smart_resize_min_pixels, smart_resize_max_pixels)
                    res_s = (res_w, res_h)
                    template_pil_resized_for_prompt = template_pil_orig.resize(res_s, Image.Resampling.BILINEAR)
                    bbox_for_text = transform_bbox(template_bbox_orig_xyxy, orig_s, res_s, 'original_to_resized')
                    if bbox_for_text: 
                        current_prompt_template_pils_resized.append(template_pil_resized_for_prompt)
                        current_prompt_template_bboxes_for_text.append(bbox_for_text)
            
            if not current_prompt_template_pils_resized: # Final fallback if all processing failed
                logger.error(f"CRITICAL: Still no valid prompt templates for frame {i} after all fallbacks. Using first frame.")
                resized_first_pil_for_prompt = first_frame_pil.resize(resized_size_first, Image.Resampling.BILINEAR)
                current_prompt_template_pils_resized = [resized_first_pil_for_prompt]
                current_prompt_template_bboxes_for_text = [first_frame_bbox_for_text_prompt]

            current_search_frame_path = seq.frames[i]
            search_pil_orig = Image.open(current_search_frame_path).convert("RGB")
            original_size_search = search_pil_orig.size
            resized_h_search, resized_w_search = smart_resize(
                original_size_search[1], original_size_search[0],
                smart_resize_factor, smart_resize_min_pixels, smart_resize_max_pixels
            )
            resized_size_search = (resized_w_search, resized_h_search)
            search_pil_resized_for_prompt = search_pil_orig.resize(resized_size_search, Image.Resampling.BILINEAR)

            messages = build_rft_input_messages(
                exp_str, 
                current_prompt_template_pils_resized, 
                search_pil_resized_for_prompt, 
                current_prompt_template_bboxes_for_text
            )
            text_prompt_with_placeholders = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            all_images_for_sample = current_prompt_template_pils_resized + [search_pil_resized_for_prompt]
            inputs = processor(
                text=[text_prompt_with_placeholders], images=all_images_for_sample,
                return_tensors="pt", padding=True,
            )
            inputs = inputs.to(model.device)

            gen_config = GenerationConfig(
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id
            )
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, generation_config=gen_config)
            
            input_token_len = inputs["input_ids"].shape[1]
            generated_text_ids = generated_ids[:, input_token_len:]
            response_text = processor.batch_decode(generated_text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            
            predicted_bbox_resized_xyxy = extract_single_bbox(response_text)
            pred_bbox_xywh_final = [0,0,0,0] 
            current_pred_bbox_original_xyxy_for_history = None

            if predicted_bbox_resized_xyxy:
                temp_pred_bbox_original_xyxy = transform_bbox(
                    predicted_bbox_resized_xyxy, resized_size_search, original_size_search,  'resized_to_original'
                )
                if temp_pred_bbox_original_xyxy and \
                   not (temp_pred_bbox_original_xyxy[2] <= temp_pred_bbox_original_xyxy[0] or \
                        temp_pred_bbox_original_xyxy[3] <= temp_pred_bbox_original_xyxy[1]):
                    pred_bbox_xywh_final = convert_bbox_xyxy_to_xywh(temp_pred_bbox_original_xyxy)
                    current_pred_bbox_original_xyxy_for_history = temp_pred_bbox_original_xyxy
            
            with open(predictions_file_path, "a") as f:
                f.write(f"{pred_bbox_xywh_final[0]},{pred_bbox_xywh_final[1]},{pred_bbox_xywh_final[2]},{pred_bbox_xywh_final[3]}\n")

            if save_visualize:
                img_draw = draw_bbox_on_image(search_pil_orig, current_pred_bbox_original_xyxy_for_history)
                img_draw.save(os.path.join(seq_output_dir, f"frame_{i:04d}_pred.jpg"))

            # Update history with current search frame's path and its predicted bbox in original coords
            # If prediction failed, current_pred_bbox_original_xyxy_for_history will be None
            all_history_frame_paths, all_history_bboxes_orig_xyxy = update_history_indexed(
                all_history_frame_paths, all_history_bboxes_orig_xyxy,
                i, current_search_frame_path, current_pred_bbox_original_xyxy_for_history
            )

        results_summary.append({'sequence': seq.name, 'output_file': predictions_file_path})
        logger.info(f"Process {rank} - Finished sequence {seq.name}. Predictions saved to {predictions_file_path}")
        
    return results_summary

def split_sequences(seq_list_all_names, rank, world_size):
    if not seq_list_all_names: return []
    n_total = len(seq_list_all_names)
    n_per_rank = math.ceil(n_total / world_size)
    start_idx = rank * n_per_rank
    end_idx = min((rank + 1) * n_per_rank, n_total)
    return seq_list_all_names[start_idx:end_idx] if start_idx < n_total else []

def run_process_wrapper(rank, world_size, args):
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): torch.cuda.set_device(rank)
    
    logger.info(f"Process {rank} started on device {device}")
    model, tokenizer, processor = load_model_and_processor(args.model_path, device=device)
    
    all_sequence_names = [seq.name for seq in get_dataset(args.dataset_name)]
    sequences_for_this_rank = [args.sequence] if args.sequence and rank == 0 else \
                              split_sequences(all_sequence_names, rank, world_size) if not args.sequence else []

    if not sequences_for_this_rank:
        logger.info(f"Process {rank} has no sequences to process.")
        return []

    logger.info(f"Process {rank} will process sequences: {sequences_for_this_rank}")
    
    return evaluate_tracking_rft(
        model, tokenizer, processor,
        sequences_names=sequences_for_this_rank,
        dataset_name=args.dataset_name,
        save_visualize=args.save_vis,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        rank=rank,
        smart_resize_factor=args.smart_resize_factor,
        smart_resize_min_pixels=args.smart_resize_min_pixels,
        smart_resize_max_pixels=args.smart_resize_max_pixels,
        gap_list=args.gap_list
    )

def main():
    parser = argparse.ArgumentParser(description="RFT Model Single Object Tracking Inference (Simplified with Gaps & Memory Optimization)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned RFT model")
    parser.add_argument("--dataset_name", type=str, default="OTB_lang", help="Dataset name")
    parser.add_argument("--sequence", type=str, default=None, help="Specific sequence name (optional)")
    parser.add_argument("--output_dir", type=str, default="rft_tracking_results_gaps_memopt", help="Output directory")
    parser.add_argument("--save_vis", type=lambda x: (str(x).lower() == 'true'), default=False, help="Save visualizations (true/false)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max new tokens for generation") # Reduced default for tracking
    parser.add_argument("--smart_resize_factor", type=int, default=DEFAULT_FACTOR, help="Factor for smart resize")
    parser.add_argument("--smart_resize_min_pixels", type=int, default=DEFAULT_MIN_PIXELS, help="Min pixels for smart resize")
    parser.add_argument("--smart_resize_max_pixels", type=int, default=DEFAULT_MAX_PIXELS, help="Max pixels for smart resize")
    parser.add_argument("--gap_list", type=int, nargs='+', default=None, help="List of gaps for selecting template frames (e.g., --gap_list 1 10). If None, uses default [1, 10].")
    parser.add_argument("--single_process", action="store_true", help="Force single process")
    args = parser.parse_args()

    if args.gap_list is None:
        args.gap_list = [1, 5] 
        logger.info(f"Using default gap_list: {args.gap_list}")

    os.makedirs(args.output_dir, exist_ok=True)
    use_multiprocessing = torch.cuda.device_count() > 1 and not args.single_process
    
    if use_multiprocessing:
        world_size = torch.cuda.device_count()
        logger.info(f"Detected {world_size} GPUs, enabling multi-process evaluation.")
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        with Pool(world_size) as pool:
            func = functools.partial(run_process_wrapper, world_size=world_size, args=args)
            results_list_of_lists = pool.map(func, range(world_size))
        all_results_summary = [item for sublist in results_list_of_lists for item in sublist]
    else:
        logger.info("Using single process evaluation.")
        rank = 0
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model, tokenizer, processor = load_model_and_processor(args.model_path, device=device)
        all_sequence_names = [seq.name for seq in get_dataset(args.dataset_name)]
        sequences_to_run = [args.sequence] if args.sequence else all_sequence_names
        if not sequences_to_run:
            logger.info("No sequences selected.")
            return
        all_results_summary = evaluate_tracking_rft(
            model, tokenizer, processor, sequences_names=sequences_to_run,
            dataset_name=args.dataset_name, save_visualize=args.save_vis,
            output_dir=args.output_dir, max_new_tokens=args.max_new_tokens, rank=rank,
            smart_resize_factor=args.smart_resize_factor,
            smart_resize_min_pixels=args.smart_resize_min_pixels,
            smart_resize_max_pixels=args.smart_resize_max_pixels,
            gap_list=args.gap_list
        )

    summary_file = os.path.join(args.output_dir, f"tracking_summary_{args.dataset_name}.json")
    with open(summary_file, "w") as f:
        json.dump(all_results_summary, f, indent=2)
    logger.info(f"Overall tracking summary saved to {summary_file}")

if __name__ == "__main__":
    main()