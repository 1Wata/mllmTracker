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
        # Last rank gets any remaining sequences
        return seq_list[rank*n_per_rank:]
    else:
        return seq_list[rank*n_per_rank:(rank+1)*n_per_rank]

# Custom single object tracking inference function
def evaluate_tracking(model, tokenizer, processor, sequences=None, dataset_name='lasot',
                      save_visualize=False, max_new_tokens=2048, rank=0, max_gap=60):
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
        max_gap: Maximum frame gap for processing
        
    Returns:
        Tracking results
    """
    # Load dataset - returns a SequenceList
    dataset = get_dataset(dataset_name)
    results = []

    # Filter sequences if specified
    if sequences:
        filtered_dataset = []
        for seq_name in sequences:
            try:
                # Try to get sequence by name
                seq = dataset[seq_name]
                filtered_dataset.append(seq)
            except IndexError:
                # Sequence not found in dataset
                logger.warning(f"Sequence {seq_name} not found in dataset")
        
        # If we found any sequences, replace dataset with filtered version
        if filtered_dataset:
            dataset = SequenceList(filtered_dataset)
        else:
            logger.warning(f"None of the specified sequences found in dataset")
            return []

    logger.info(f"Process {rank} - Processing {len(dataset)} sequences")


    for seq in tqdm(dataset, desc=f"Process {rank} - Tracking progress"):
        seq_results = []
        previous_imgs = list()
        previous_normed_bboxs = list()

        first_frame = Image.open(seq.frames[0])
        init_info = seq.init_info()
        first_frame_bbox = init_info.get('init_bbox')
        exp_str = init_info.get('init_text_description')
        
        first_frame_normed_bbox = normalize_bbox_xyhw(first_frame_bbox, first_frame.size[0], first_frame.size[1])
        first_frame_drawed = draw_normed_bbox(first_frame, first_frame_normed_bbox)
        print(f"Process {rank} - First frame bbox: {first_frame_normed_bbox}")
        # Create directory for each sequence
        os.makedirs(f"tracking_results/{seq.name}", exist_ok=True)
        first_frame_drawed.save(f"tracking_results/{seq.name}/frame_0000.jpg")
        
        # Select frames to process based on max_gap

        frames_to_process = list(range(1, len(seq.frames)))
        
        for i in frames_to_process:
            # 不再简单地使用 [first_frame] + previous_imgs
            # 而是采用更符合训练策略的采样方法
            
            # 始终保留第一帧作为长期参考
            template_imgs = [first_frame]
            template_bboxs = [first_frame_normed_bbox]
            
            # 从历史帧中有策略地选择模板帧
            if len(previous_imgs) > 0:
                # 跳过最近的几帧，避免模板帧和搜索帧间隔太小
                skip_recent = min(3, len(previous_imgs) // 2)
                
                # 从剩余历史中选择帧，模拟训练时的间隔采样
                if len(previous_imgs) > skip_recent:
                    # 按照一定间隔选择历史帧
                    indices = np.linspace(0, len(previous_imgs) - skip_recent - 1, 
                                         min(3, len(previous_imgs) - skip_recent), 
                                         dtype=int)
                    
                    for idx in indices:
                        template_imgs.append(previous_imgs[idx])
                        template_bboxs.append(previous_normed_bboxs[idx])
            

            search_img = Image.open(seq.frames[i])
            messages = build_input_message(dataset_name=dataset_name, exp_str=exp_str, 
                                          template_imgs=template_imgs,
                                          search_img=search_img, 
                                          normalized_template_bboxes=template_bboxs)
            print("message before apply_chat_template")
            print(messages)
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print("message after apply_chat_template")
            print(text)
            image_inputs, _ = process_vision_info([messages])



            inputs = processor(
                text=[text],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
            response = generate_output(model, tokenizer, inputs, max_new_tokens)
            # Extract bounding box
            print(response)
            predicted_box = extract_bbox_from_response(response)

            
            if predicted_box:
                search_img_drawed = draw_normed_bbox(search_img, predicted_box)
                previous_imgs, previous_normed_bboxs = update_previous_imgs_bboxs(
                    previous_imgs, search_img, previous_normed_bboxs,
                    predicted_box, cfg.DATA.PREV_TEMPLATE.NUMBER + 1, frame_idx=i
                )
                
                # Save the search image with bounding box
                if save_visualize:
                    save_path = f"tracking_results/{seq.name}/frame_{i:04d}.jpg"
                    search_img_drawed.save(save_path)
                
                # Save the predicted bounding box to file
                gt_file_path = f"tracking_results/{seq.name}/predictions.txt"
                with open(gt_file_path, "a" if i > 0 else "w") as f:
                    x1, y1, x2, y2 = predicted_box
                    x1, y1, x2, y2 = unnormalize_bbox([x1, y1, x2, y2], search_img.size[0], search_img.size[1])
                    # Save in format: [x, y, width, height]
                    width = x2 - x1
                    height = y2 - y1
                    f.write(f"{x1},{y1},{width},{height}\n")
                    
                seq_results.append({
                    'frame_id': i,
                    'bbox': [x1, y1, width, height]
                })
                logger.info(f"Process {rank} - Saved frame {i} for sequence {seq.name}")
            else:
                # If prediction failed, write a placeholder or null entry
                gt_file_path = f"tracking_results/{seq.name}/predictions.txt"
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
        # 获取序列名称列表而不是序列对象
        sequence_names = [seq.name for seq in dataset]
        # 基于序列名称进行分配
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
                              predicted_bbox, max_length, frame_idx=None):
    """
    更新历史帧列表，使用更符合训练数据分布的策略
    
    Args:
        previous_imgs: 历史帧列表
        frame: 当前帧
        previous_bboxs: 历史边界框列表
        predicted_bbox: 当前预测的边界框
        max_length: 最大历史长度
        frame_idx: 当前帧索引，用于采样决策
    """
    # 帧采样频率控制 - 不是每一帧都保存
    sampling_rate = 3  # 可调整的采样率
    
    # 基于帧索引决定是否添加到历史
    if frame_idx is not None and frame_idx % sampling_rate != 0:
        # 如果不满足采样条件，不更新历史
        return previous_imgs, previous_bboxs
    
    # 如果历史长度未达到最大值，直接添加
    if len(previous_imgs) < max_length:
        previous_imgs.append(frame)
        previous_bboxs.append(predicted_bbox)
    else:
        # 不是简单的FIFO，而是保持更有代表性的时间分布
        # 例如: 保留最早、中间和最近的帧，删除其他帧
        if len(previous_imgs) > 3:
            indices_to_keep = [0, len(previous_imgs)//2, -1]  # 保留首、中、尾
            previous_imgs = [previous_imgs[i] for i in indices_to_keep]
            previous_bboxs = [previous_bboxs[i] for i in indices_to_keep]
            
            # 确保不超过最大长度的情况下添加当前帧
            if len(previous_imgs) < max_length:
                previous_imgs.append(frame)
                previous_bboxs.append(predicted_bbox)
        else:
            # 历史太短，使用标准FIFO
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
    # Use regex to match all occurrences of [x1, y1, x2, y2] format with decimal numbers
    pattern = r'\[(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+),\s*(\d+\.\d+|\d+)\]'
    matches = list(re.finditer(pattern, response))
    
    if matches:
        # Get the last match
        last_match = matches[-1]
        x1 = int(last_match.group(1))
        y1 = int(last_match.group(2))
        x2 = int(last_match.group(3))
        y2 = int(last_match.group(4))
        return [x1, y1, x2, y2]
    
    # Try another format with different spacing
    pattern = r'\[\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*\]'
    matches = list(re.finditer(pattern, response))
    
    if matches:
        # Get the last match
        last_match = matches[-1]
        x1 = int(last_match.group(1))
        y1 = int(last_match.group(2))
        x2 = int(last_match.group(3))
        y2 = int(last_match.group(4))
        return [x1, y1, x2, y2]
    
    logger.warning(f"Failed to extract bounding box from response: {response}")
    return None



def main():
    parser = argparse.ArgumentParser(description="Visual Language Model Single Object Tracking Evaluation")
    parser.add_argument("--model_path", type=str, default="/data1/lihaobo/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/tracking_large-3",
                        help="Model path")
    parser.add_argument("--dataset_name", type=str, default="lasot",
                        help="Dataset name")
    parser.add_argument("--sequence", type=str, default='person-1',
                        help="Specific sequence name (optional)")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum new tokens to generate")
    parser.add_argument("--init_frames", type=int, default=1,
                        help="Number of initial reference frames")
    parser.add_argument("--context_window", type=int, default=2,
                        help="Context window size for prediction")
    parser.add_argument("--visualize", action="store_true", 
                        help="Visualize tracking results")
    parser.add_argument("--save_vis", default=True, 
                        help="Save visualization results")
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Visualization results directory")
    parser.add_argument("--single_process", action="store_true",
                        help="Force single process execution even with multiple GPUs")
    
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs("tracking_results", exist_ok=True)
    
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
            save_visualize=args.save_vis,
            dataset_name=args.dataset_name,
            max_new_tokens=args.max_new_tokens
        )
        
        # Save results
        result_filename = f"tracking_results/results_{args.dataset_name}"
        if args.sequence:
            result_filename += f"_{args.sequence}"
        result_filename += ".json"
        
        with open(result_filename, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Single process evaluation complete, results saved to {result_filename}")


if __name__ == "__main__":
    main()


