import os
import re
import json

import torch
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoTokenizer
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          StoppingCriteria, StoppingCriteriaList)
from transformers.generation import GenerationConfig

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
import logging
import argparse
import functools
import multiprocessing as mp
from multiprocessing import Pool

from easydict import EasyDict
import yaml

from evaluation.datasets import get_dataset, SequenceList
from utils.build_message import build_input_message
from utils.utils import normalize_bbox_xyhw, draw_normed_bbox, unnormalize_bbox
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import time

cfg = EasyDict(
    yaml.safe_load(
        open(os.path.join(os.path.dirname(__file__), "dataset_config.yaml"))
    )
)

# Load model and tokenizer
def load_model(model_path, device="auto"):
    logger.info(f"Loading model from {model_path} to device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device
    )
    model.eval()
    return model, tokenizer, processor


# Split sequences for multi-process
def split_sequences(seq_list, rank, world_size):
    if not seq_list:
        return None
    
    n_per_rank = len(seq_list) // world_size
    if rank == world_size - 1:
        # Last rank gets any remaining sequencesf
        return seq_list[rank*n_per_rank:]
    else:
        return seq_list[rank*n_per_rank:(rank+1)*n_per_rank]

# Custom single object tracking inference function
def evaluate_tracking(model, tokenizer, processor, sequences=None, dataset_name='lasot',
                      save_visualize=False, output_dir=None, max_new_tokens=2048, rank=0):
    """
    Single object tracking evaluation function
    
    Args:
        model: Model
        tokenizer: Tokenizer
        processor: Processor
        sequences: List of specific sequence names to process
        dataset_name: Dataset name
        max_new_tokens: Maximum number of new tokens to generate
        rank: Process rank for logging
        
    Returns:
        Tracking results
    """
    if output_dir is None:
        output_dir = f"tracking_results/{dataset_name}"
    # Load dataset - returns a SequenceList
    dataset = get_dataset(dataset_name)
    results = []

    # Filter sequences if specified
    if sequences:
        filtered_dataset = []
        for seq_name in sequences:

            seq = dataset[seq_name]
            filtered_dataset.append(seq)

        

        if filtered_dataset:
            dataset = SequenceList(filtered_dataset)
        else:
            logger.warning(f"None of the specified sequences found in dataset")
            return []

    logger.info(f"Process {rank} - Processing {len(dataset)} sequences")


    image_open_times = []
    generate_output_times = []

    for seq in tqdm(dataset, desc=f"Process {rank} - Tracking progress"):
        seq_results = []
        previous_imgs = list()
        previous_normed_bboxs = list()

        first_frame = Image.open(seq.frames[0])
        init_info = seq.init_info()
        first_frame_bbox = init_info.get('init_bbox')
        gt_file_path = f"{output_dir}/{seq.name}.txt"
        with open(gt_file_path, "w") as f:
            x, y, w, h = first_frame_bbox
            f.write(f"{x},{y},{w},{h}\n")
            
        exp_str = init_info.get('init_text_description')
        
        first_frame_normed_bbox = normalize_bbox_xyhw(first_frame_bbox, first_frame.size[0], first_frame.size[1])
        first_frame_drawed = draw_normed_bbox(first_frame, first_frame_normed_bbox)

        # Create directory for each sequence
        # os.makedirs(f"tracking_results/{seq.name}", exist_ok=True)
        if save_visualize:
            os.makedirs(f"{output_dir}/{seq.name}", exist_ok=True)
            first_frame_drawed.save(f"{output_dir}/{seq.name}/frame_0000.jpg")
        for i, frame in enumerate(seq.frames[1:], start=1):
            template_imgs = [first_frame] + previous_imgs
            normed_bboxs = [first_frame_normed_bbox] + previous_normed_bboxs


            start_time = time.time()
            search_img = Image.open(frame)
            image_open_times.append(time.time() - start_time)
            messages = build_input_message(dataset_name=dataset_name, exp_str=exp_str, template_imgs=template_imgs,
                                           search_img=search_img, normalized_template_bboxes=normed_bboxs)
            # print("message before apply_chat_template")
            # print(messages)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # print("message after apply_chat_template")
            # print(text)
            image_inputs, _ = process_vision_info([messages])



            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            start_time = time.time()
            response = generate_output(model, tokenizer, inputs, max_new_tokens)
            generate_output_times.append(time.time() - start_time)
            # Extract bounding box
            # print(response)
            predicted_box = extract_bbox_from_response(response)

            
            if predicted_box:
                search_img_drawed = draw_normed_bbox(search_img, predicted_box)
                previous_imgs, previous_normed_bboxs = update_previous_imgs_bboxs(
                    previous_imgs, search_img, previous_normed_bboxs,
                    predicted_box, cfg.DATA.TEMPLATE.NUMBER
                )
                
                # Save the search image with bounding box
                if save_visualize:
                    save_path = f"{output_dir}/{seq.name}/frame_{i:04d}.jpg"
                    search_img_drawed.save(save_path)
                
                # Save the predicted bounding box to file
                
                with open(gt_file_path, "a" if i > 0 else "w") as f:
                    x1, y1, x2, y2 = predicted_box
                    x1, y1, width, height = unnormalize_bbox([x1, y1, x2, y2], search_img.size[0], search_img.size[1])
                    # Save in format: [x, y, width, height]
                    f.write(f"{x1},{y1},{width},{height}\n")
                    
                seq_results.append({
                    'frame_id': i,
                    'bbox': [x1, y1, width, height]
                })
                logger.info(f"Process {rank} - Saved frame {i} for sequence {seq.name}")
            else:
                # If prediction failed, write a placeholder or null entry
                # gt_file_path = f"tracking_results/{seq.name}/predictions.txt"
                with open(gt_file_path, "a" if i > 0 else "w") as f:
                    f.write("0,0,0,0\n")
                seq_results.append({
                    'frame_id': i,
                    'bbox': [0, 0, 0, 0]
                })
                logger.info(f"Process {rank} - Failed to extract bbox for frame {i} in sequence {seq.name}")
        
        results.append({
            'sequence_name': seq.name,
            'frames': seq_results
        })
        

    
    avg_image_open_time = sum(image_open_times) / len(image_open_times) if image_open_times else 0
    avg_generate_output_time = sum(generate_output_times) / len(generate_output_times) if generate_output_times else 0
    print(f"Process {rank} - Average image open time: {avg_image_open_time:.4f}s")
    print(f"Process {rank} - Average generate output time: {avg_generate_output_time:.4f}s")
    return results

def run_process(rank, world_size, args):
    """
    Single process execution function
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Arguments
        
    Returns:
        Evaluation results for this process
    """
    # Set specific GPU device
    if torch.cuda.is_available():
        device = f"cuda:{rank}"
        torch.cuda.set_device(rank)
    else:
        device = "cpu"
    
    logger.info(f"Process {rank} started on device {device}")
    
    # Load model on specific device
    model, tokenizer, processor = load_model(args.model_path, device=device)
    
    # Get sequences for this process
    if args.sequence:
        # Single sequence mode - only process this on rank 0
        sequences = [args.sequence] if rank == 0 else []
    else:
        # Multi-sequence mode - distribute sequences across processes
        dataset = get_dataset(args.dataset_name)
        sequence_names = [seq.name for seq in dataset]
        sequences = split_sequences(sequence_names, rank, world_size)
    
    if not sequences:
        logger.info(f"Process {rank} has no sequences to process")
        return []
    
    logger.info(f"Process {rank} will process {len(sequences)} sequences")
    
    # Run evaluation
    results = evaluate_tracking(
        model,
        tokenizer,
        processor,
        sequences=sequences,
        dataset_name=args.dataset_name,
        max_new_tokens=args.max_new_tokens,
        rank=rank
    )
    
    # Save intermediate results from this process
    with open(f"tracking_results/results_rank_{rank}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def update_previous_imgs_bboxs(previous_imgs, frame, previous_bboxs,
                               predicted_bbox, max_length):
    """
    Update context frame list
    
    Args:
        previous_imgs: Context frames list
        frame: New frame image
        max_length: Maximum length of context
    """
    if len(previous_imgs) < max_length:
        previous_imgs.append(frame)
        previous_bboxs.append(predicted_bbox)
    else:
        previous_imgs.pop(0)
        previous_imgs.append(frame)
        previous_bboxs.pop(0)
        previous_bboxs.append(predicted_bbox)
    
    return previous_imgs, previous_bboxs



def generate_output(model, tokenizer, inputs, max_new_tokens=2048, full_conversation=True):
    """
    Generate output using model.generate()
    
    Args:
        model: The model to generate with
        tokenizer: Tokenizer for decoding
        inputs: Preprocessed inputs
        max_new_tokens: Maximum number of new tokens to generate
        full_conversation: If True, return the full input and output; if False, return only the response
        
    Returns:
        Decoded response string
    """
    # Move inputs to model device
    inputs = inputs.to(model.device)
    
    # Use model.generate() for more efficient generation
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Use greedy decoding instead of sampling
            temperature=None,
        )
    
    if full_conversation:
        # Return the full conversation including the input
        full_response = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        return full_response[0]
    else:
        # Extract only the newly generated tokens (excluding prompt)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        # Decode the generated tokens
        response = tokenizer.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return response[0]  # Return the first (and only) decoded response

def extract_bbox_from_response(response):
    """
    Extract the last bounding box coordinates from model response
    """

    patterns = [
        r'\[(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+)\]',
        r'\[\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*\]'
    ]
    
    # 尝试每种模式
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
    
    # 如果所有模式都未匹配成功
    logger.warning(f"Failed to extract bounding box from response: {response}")
    return None



def main():
    parser = argparse.ArgumentParser(description="Visual Language Model Single Object Tracking Evaluation")
    parser.add_argument("--model_path", type=str, default="/data1/lihaobo/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/tracking_large-2",
                        help="Model path")
    parser.add_argument("--dataset_name", type=str, default="OTB_lang",
                        help="Dataset name")
    parser.add_argument("--output_dir", type=str, default="/data1/lihaobo/tracking/output_dir",
                        help="Output directory")
    parser.add_argument("--sequence", type=str, default='Biker',
                        help="Specific sequence name (optional)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum new tokens to generate")
    # parser.add_argument("--init_frames", type=int, default=1,
                        # help="Number of initial reference frames")
    parser.add_argument("--context_window", type=int, default=2,
                        help="Context window size for prediction")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize tracking results")
    parser.add_argument("--save_vis", default=False, 
                        help="Save visualization results")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Visualization results directory")
    parser.add_argument("--single_process", action="store_true",
                        help="Force single process execution even with multiple GPUs")
    
    args = parser.parse_args()
    
    # Create results directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Check if multi-processing can be used
    use_multiprocessing = torch.cuda.device_count() > 1 and not args.single_process
    
    if use_multiprocessing:
        logger.info(f"Detected {torch.cuda.device_count()} GPUs, enabling multi-process evaluation")
        mp.set_start_method('spawn')
        
        world_size = torch.cuda.device_count()
        with Pool(world_size) as pool:
            func = functools.partial(run_process, world_size=world_size, args=args)
            results_list = pool.map(func, range(world_size))
        
        # Merge results from all processes
        all_results = []
        for result in results_list:
            all_results.extend(result)
        
        # Save merged results
        with open(f"tracking_results/combined_results_{args.dataset_name}.json", "w") as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Multi-process evaluation complete, results saved to tracking_results/combined_results_{args.dataset_name}.json")
    else:
        logger.info("Using single process evaluation")
        if args.sequence:
            logger.info(f"Evaluating single sequence: {args.sequence}")
        else:
            logger.info(f"Evaluating all sequences in dataset {args.dataset_name}")
            
        model, tokenizer, processor = load_model(args.model_path)
        
        sequences = [args.sequence] if args.sequence else None
        results = evaluate_tracking(
            model,
            tokenizer,
            processor,
            sequences=sequences,
            output_dir=args.output_dir,
            save_visualize=args.save_vis,
            dataset_name=args.dataset_name,
            max_new_tokens=args.max_new_tokens
        )
        
        # Save results
        result_filename = f"tracking_results/{args.dataset_name}"
        if args.sequence:
            result_filename += f"_{args.sequence}"
        result_filename += ".txt"
        
        with open(result_filename, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Single process evaluation complete, results saved to {result_filename}")


if __name__ == "__main__":
    main()