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

from evaluation.datasets import get_dataset, SequenceList # Keep existing import
from utils.build_message import build_input_message # Keep existing import, maybe create a new one
from utils.utils import normalize_bbox_xyhw, draw_normed_bbox, unnormalize_bbox # Keep existing imports
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

    image_max_pixels = 65536
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=image_max_pixels)
    
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

# --- New function to build the one-turn message for inference ---
def build_oneturn_inference_message(dataset_name, exp_str, template_img, template_bbox_norm, search_imgs):
    """
    Builds a single-turn message for predicting multiple frames based on one template frame.

    Args:
        dataset_name (str): The name of the dataset.
        exp_str (str): The category/object name or description.
        template_img (PIL.Image): The template frame image.
        template_bbox_norm (list[float]): Normalized bounding box [x1, y1, x2, y2] for the template frame.
        search_imgs (list[PIL.Image]): List of search frame images.

    Returns:
        list[dict]: A list containing the message structure for the model input.
    """
    messages = []
    image_placeholders = ["<image>"] * (1 + len(search_imgs)) # One for template, others for search

    # Format the template bounding box string
    normalized_ref_bbox_str = (f" The object in this frame is at normalized coordinates [{template_bbox_norm[0]:.3f}, {template_bbox_norm[1]:.3f}, "
                               f"{template_bbox_norm[2]:.3f}, {template_bbox_norm[3]:.3f}].")

    # Build the user message
    user_content = (f"You are an AI assistant for single object tracking. "
                   f"This image ({image_placeholders[0]}) shows the object of interest: '{exp_str}'."
                   f"{normalized_ref_bbox_str} ")

    # Add references to all search frames
    user_content += f"Please track this object across the following {len(search_imgs)} frames:"
    for i in range(len(search_imgs)):
        user_content += f" ({image_placeholders[i+1]})" # Placeholder for each search image

    # Add output format instructions
    user_content += (f". For each frame (starting from Frame 1 for the first search image), reply in exactly this format:\n"
                    f"Frame 1: [x1, y1, x2, y2] or \"not visible\"\n"
                    f"Frame 2: [x1, y1, x2, y2] or \"not visible\"\n"
                    f"... and so on for all {len(search_imgs)} frames.")

    messages.append({"role": "user", "content": user_content, "images": [template_img] + search_imgs})

    return messages

# --- Modified function to extract bounding boxes for all frames ---
def extract_all_bboxes_from_response(response, num_frames):
    """
    Extracts bounding box coordinates for multiple frames from the model response.
    Expects format like:
    Frame 1: [x1, y1, x2, y2]
    Frame 2: not visible
    ...

    Args:
        response (str): The full response string from the model.
        num_frames (int): The expected number of frames (excluding the template).

    Returns:
        list[list[float] or None]: A list containing the bounding box [x1, y1, x2, y2] for each frame,
                                   or None if not found or "not visible".
    """
    predicted_boxes = [None] * num_frames
    lines = response.strip().split('\n')

    bbox_pattern = r'\[\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*,\s*(\d+\.\d+|\d+)\s*\]'
    frame_pattern = r'Frame\s+(\d+):'

    current_frame_index = -1

    for line in lines:
        line = line.strip()
        frame_match = re.search(frame_pattern, line)
        if frame_match:
            try:
                # Frame index in the response (1-based)
                frame_num = int(frame_match.group(1))
                # Convert to 0-based index for the list
                current_frame_index = frame_num - 1
                if current_frame_index < 0 or current_frame_index >= num_frames:
                    logger.warning(f"Parsed frame index {frame_num} out of expected range (1-{num_frames}) in line: {line}")
                    current_frame_index = -1 # Reset if out of bounds
                    continue

                bbox_match = re.search(bbox_pattern, line)
                if bbox_match:
                     # Extract and round coordinates
                    x1 = round(float(bbox_match.group(1)), 3) # Using 3 decimal places like normalization
                    y1 = round(float(bbox_match.group(2)), 3)
                    x2 = round(float(bbox_match.group(3)), 3)
                    y2 = round(float(bbox_match.group(4)), 3)
                    # Basic validation: ensure coordinates are within [0, 1] and x1 < x2, y1 < y2
                    if 0 <= x1 <= 1 and 0 <= y1 <= 1 and 0 <= x2 <= 1 and 0 <= y2 <= 1 and x1 < x2 and y1 < y2:
                         predicted_boxes[current_frame_index] = [x1, y1, x2, y2]
                    else:
                        logger.warning(f"Invalid bbox coordinates extracted for frame {frame_num}: {[x1, y1, x2, y2]}. Setting to None.")
                        predicted_boxes[current_frame_index] = None

                elif "not visible" in line.lower():
                    predicted_boxes[current_frame_index] = None
                # else: # Handle cases where frame number is found but no valid bbox or "not visible" text
                    # logger.warning(f"Could not parse bbox or 'not visible' for frame {frame_num} in line: {line}")
                    # predicted_boxes[current_frame_index] = None # Ensure it's None if parsing fails after finding frame number

            except ValueError:
                logger.warning(f"Could not parse frame number from line: {line}")
                current_frame_index = -1 # Reset if frame number parsing fails
            except IndexError:
                 logger.warning(f"Frame index {current_frame_index+1} is out of bounds for predicted_boxes list (size {num_frames}).")
                 current_frame_index = -1 # Reset if index is invalid

        # Fallback: If a line contains just a bbox without "Frame X:", try to assign it sequentially?
        # This is less robust. Let's rely on the "Frame X:" prefix for now.
        # else:
        #     bbox_match = re.search(bbox_pattern, line)
        #     if bbox_match and any(p is None for p in predicted_boxes):
        #         # Find the first None slot and fill it? Risky.
        #         pass


    # Check if the number of extracted boxes matches expected frames
    # num_found = sum(1 for box in predicted_boxes if box is not None)
    # if num_found != num_frames and len(lines) < num_frames: # Simple check, might need refinement
    #     logger.warning(f"Expected {num_frames} bounding boxes, but parsed {num_found} from response: {response}")

    return predicted_boxes


# --- Modified evaluation function ---
def evaluate_tracking(model, tokenizer, processor, sequences=None, dataset_name='lasot',
                      save_visualize=False, output_dir=None, max_new_tokens=2048, rank=0):
    """
    Single object tracking evaluation function using one template frame to predict all subsequent frames.

    Args:
        model: Model
        tokenizer: Tokenizer
        processor: Processor
        sequences: List of specific sequence names to process
        dataset_name: Dataset name
        max_new_tokens: Maximum number of new tokens to generate per sequence prediction
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
            try:
                seq = dataset[seq_name]
                filtered_dataset.append(seq)
            except KeyError:
                 logger.warning(f"Sequence '{seq_name}' not found in dataset '{dataset_name}'. Skipping.")

        if filtered_dataset:
            dataset = SequenceList(filtered_dataset)
        else:
            logger.warning(f"None of the specified sequences found in dataset")
            return []

    logger.info(f"Process {rank} - Processing {len(dataset)} sequences")

    image_open_times = []
    generate_output_times = []
    processing_times = []

    for seq in tqdm(dataset, desc=f"Process {rank} - Tracking progress"):
        seq_start_time = time.time()
        seq_results = []
        num_frames = len(seq.frames)
        if num_frames < 2:
            logger.warning(f"Sequence {seq.name} has less than 2 frames. Skipping.")
            continue

        # --- Load first frame (template) ---
        try:
            first_frame = Image.open(seq.frames[0]).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to open first frame for sequence {seq.name}: {e}. Skipping sequence.")
            continue

        init_info = seq.init_info()
        first_frame_bbox = init_info.get('init_bbox') # [x, y, w, h]
        if not first_frame_bbox:
             logger.error(f"Missing initial bounding box for sequence {seq.name}. Skipping sequence.")
             continue

        exp_str = init_info.get('init_text_description', 'the object') # Use default if missing

        # --- Prepare output file and write first frame GT ---
        gt_file_path = f"{output_dir}/{seq.name}.txt"
        os.makedirs(os.path.dirname(gt_file_path), exist_ok=True) # Ensure directory exists
        with open(gt_file_path, "w") as f:
            x, y, w, h = first_frame_bbox
            f.write(f"{x},{y},{w},{h}\n")

        # --- Normalize first frame bbox to [x1, y1, x2, y2] format ---
        first_frame_normed_bbox = normalize_bbox_xyhw(first_frame_bbox, first_frame.size[0], first_frame.size[1])

        # --- Optional: Save visualized first frame ---
        if save_visualize:
            os.makedirs(f"{output_dir}/{seq.name}", exist_ok=True)

            first_frame_drawed = draw_normed_bbox(first_frame.copy(), first_frame_normed_bbox)
            first_frame_drawed.save(f"{output_dir}/{seq.name}/frame_0000.jpg")



        # --- Load all subsequent frames (search frames) ---
        search_imgs = []
        search_img_paths = seq.frames[1:]
        current_image_open_times = []
        for frame_path in search_img_paths:
            start_time = time.time()
            try:
                img = Image.open(frame_path).convert('RGB')
                search_imgs.append(img)
                current_image_open_times.append(time.time() - start_time)
            except Exception as e:
                logger.error(f"Failed to open search frame {frame_path} for sequence {seq.name}: {e}. Stopping processing for this sequence.")
                search_imgs = [] # Clear list to prevent processing incomplete sequence
                break # Stop processing this sequence
        image_open_times.extend(current_image_open_times)

        if not search_imgs or len(search_imgs) != num_frames - 1:
            logger.warning(f"Could not load all search frames for sequence {seq.name}. Skipping prediction.")
            # Write dummy results for remaining frames
            with open(gt_file_path, "a") as f:
                for _ in range(num_frames - 1):
                     f.write("0,0,0,0\n")
            results.append({'sequence_name': seq.name, 'frames': [{'frame_id': i+1, 'bbox': [0,0,0,0]} for i in range(num_frames - 1)]})
            continue # Skip to next sequence


        # --- Build the single-turn message ---
        messages = build_oneturn_inference_message(
            dataset_name=dataset_name,
            exp_str=exp_str,
            template_img=first_frame,
            template_bbox_norm=first_frame_normed_bbox,
            search_imgs=search_imgs
        )

        # --- Prepare inputs for the model ---
        try:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            # Note: process_vision_info expects a list of conversations
            # Our message format puts images inside the message dict, need adaptation if process_vision_info expects them separately
            # Assuming processor handles images correctly when passed directly in the message structure
            # If not, `process_vision_info` might need adjustment or direct image list passing.
            # Let's try passing images directly to the processor as intended by some newer models/processors.

            # Extract images for processor input
            all_images = messages[0]['images'] # [template_img, search_img1, search_img2, ...]

            inputs = processor(
                text=[text], # Batch size of 1
                images=all_images, # Pass list of PIL images
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)
        except Exception as e:
            logger.error(f"Failed during input processing for sequence {seq.name}: {e}. Skipping prediction.")
            # Write dummy results
            with open(gt_file_path, "a") as f:
                for _ in range(len(search_imgs)):
                     f.write("0,0,0,0\n")
            results.append({'sequence_name': seq.name, 'frames': [{'frame_id': i+1, 'bbox': [0,0,0,0]} for i in range(len(search_imgs))]})
            continue


        # --- Generate the response for all frames ---
        start_time = time.time()
        try:
            response = generate_output(model, tokenizer, inputs, max_new_tokens, full_conversation=False) # Get only the generated part
            generate_output_times.append(time.time() - start_time)
        except Exception as e:
             logger.error(f"Failed during model generation for sequence {seq.name}: {e}. Skipping prediction.")
             response = "" # Set response to empty to handle failure case below
             generate_output_times.append(time.time() - start_time) # Still record time


        # --- Extract bounding boxes for all frames from the response ---
        predicted_normed_bboxes = extract_all_bboxes_from_response(response, len(search_imgs))

        # --- Process and save results for each frame ---
        current_seq_frame_results = []
        with open(gt_file_path, "a") as f: # Append results to the file
            for i, predicted_box in enumerate(predicted_normed_bboxes):
                frame_index = i + 1 # 1-based index for frame number
                search_img = search_imgs[i]

                if predicted_box: # If a valid normalized bbox [x1, y1, x2, y2] was extracted
                    try:
                        # Unnormalize bbox
                        x, y, w, h = unnormalize_bbox(predicted_box, search_img.size[0], search_img.size[1])

                        # Save visualized frame if enabled
                        if save_visualize:
                             try:
                                search_img_drawed = draw_normed_bbox(search_img.copy(), predicted_box)
                                save_path = f"{output_dir}/{seq.name}/frame_{frame_index:04d}.jpg"
                                search_img_drawed.save(save_path)
                             except Exception as e:
                                 logger.warning(f"Failed to save visualized frame {frame_index} for {seq.name}: {e}")


                        # Write to file
                        f.write(f"{x},{y},{w},{h}\n")
                        current_seq_frame_results.append({
                            'frame_id': frame_index,
                            'bbox': [x, y, w, h]
                        })
                        # logger.info(f"Process {rank} - Saved frame {frame_index} for sequence {seq.name}")
                    except Exception as e:
                         logger.error(f"Error processing/saving valid prediction for frame {frame_index}, sequence {seq.name}: {e}. Saving as [0,0,0,0].")
                         f.write("0,0,0,0\n")
                         current_seq_frame_results.append({'frame_id': frame_index, 'bbox': [0, 0, 0, 0]})

                else: # If prediction failed or object not visible
                    f.write("0,0,0,0\n")
                    current_seq_frame_results.append({
                        'frame_id': frame_index,
                        'bbox': [0, 0, 0, 0]
                    })
                    # logger.info(f"Process {rank} - Failed to extract/find bbox for frame {frame_index} in sequence {seq.name}")

        results.append({
            'sequence_name': seq.name,
            'frames': current_seq_frame_results
        })
        processing_times.append(time.time() - seq_start_time)
        logger.info(f"Process {rank} - Finished sequence {seq.name} in {time.time() - seq_start_time:.2f}s")


    # --- Log average times ---
    avg_image_open_time = sum(image_open_times) / len(image_open_times) if image_open_times else 0
    avg_generate_output_time = sum(generate_output_times) / len(generate_output_times) if generate_output_times else 0
    avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
    print(f"Process {rank} - Average image open time per image: {avg_image_open_time:.4f}s")
    print(f"Process {rank} - Average generate output time per sequence: {avg_generate_output_time:.4f}s")
    print(f"Process {rank} - Average total processing time per sequence: {avg_processing_time:.4f}s")
    return results

# --- run_process function remains largely the same, but calls the modified evaluate_tracking ---
def run_process(rank, world_size, args):
    """
    Single process execution function (Modified to use new evaluate_tracking)

    Args:
        rank: Process rank
        world_size: Total number of processes
        args: Arguments

    Returns:
        Evaluation results for this process
    """
    # Set specific GPU device
    if torch.cuda.is_available():
        # Ensure world_size doesn't exceed available GPUs if using CUDA
        num_gpus = torch.cuda.device_count()
        if world_size > num_gpus:
            logger.warning(f"World size ({world_size}) exceeds available GPUs ({num_gpus}). Adjusting world size.")
            world_size = num_gpus
            if rank >= world_size:
                logger.info(f"Process {rank} is redundant due to GPU limit. Exiting.")
                return [] # Exit if rank is now out of bounds

        target_gpu = rank % num_gpus # Assign GPU in round-robin if world_size > num_gpus (shouldn't happen with check above, but safe)
        device = f"cuda:{target_gpu}"
        torch.cuda.set_device(target_gpu)
    else:
        device = "cpu"
        if rank > 0:
             logger.info(f"Process {rank} running on CPU (redundant). Exiting.")
             return [] # Only rank 0 runs on CPU if no GPUs

    logger.info(f"Process {rank} started on device {device}")

    # Load model on specific device
    model, tokenizer, processor = load_model(args.model_path, device=device)

    # Get sequences for this process
    all_sequence_names = []
    if not args.sequence: # Only get full dataset if not evaluating a single sequence
        try:
            dataset = get_dataset(args.dataset_name)
            all_sequence_names = [seq.name for seq in dataset]
        except Exception as e:
            logger.error(f"Process {rank} failed to load dataset {args.dataset_name}: {e}")
            return []

    if args.sequence:
        # Single sequence mode - only process this on rank 0
        sequences_for_rank = [args.sequence] if rank == 0 else []
    else:
        # Multi-sequence mode - distribute sequences across processes
        sequences_for_rank = split_sequences(all_sequence_names, rank, world_size)

    if not sequences_for_rank:
        logger.info(f"Process {rank} has no sequences to process")
        return []

    logger.info(f"Process {rank} will process {len(sequences_for_rank)} sequences: {sequences_for_rank[:5]}...") # Log first few

    # --- Ensure output directory exists before evaluation ---
    process_output_dir = os.path.join(args.output_dir, args.dataset_name) # Subdir per dataset
    os.makedirs(process_output_dir, exist_ok=True)


    # Run evaluation
    results = evaluate_tracking(
        model,
        tokenizer,
        processor,
        sequences=sequences_for_rank,
        dataset_name=args.dataset_name,
        output_dir=process_output_dir, # Pass the specific output dir
        save_visualize=args.save_vis, # Pass save_vis flag
        max_new_tokens=args.max_new_tokens,
        rank=rank
    )

    # Save intermediate results from this process (optional, maybe remove if main merging works)
    # intermediate_results_dir = os.path.join(args.output_dir, "intermediate_results")
    # os.makedirs(intermediate_results_dir, exist_ok=True)
    # try:
    #     with open(os.path.join(intermediate_results_dir, f"results_{args.dataset_name}_rank_{rank}.json"), "w") as f:
    #         json.dump(results, f, indent=2)
    # except Exception as e:
    #      logger.error(f"Process {rank} failed to save intermediate results: {e}")


    return results


# --- Remove update_previous_imgs_bboxs function as it's no longer used ---
# def update_previous_imgs_bboxs(previous_imgs, frame, previous_bboxs,
#                                predicted_bbox, max_length):
#     ... # Removed


# --- generate_output function remains the same ---
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
    # inputs = {k: v.to(model.device) for k, v in inputs.items()} # Ensure all parts are on device

    # Use model.generate() for more efficient generation
    with torch.no_grad():
        # Use temperature=0.0 for deterministic output if do_sample=False
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0, # Set temperature to 0 for greedy decoding
            # top_p=None, # Not needed if do_sample=False
        )

    input_token_len = inputs['input_ids'].shape[1]

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
        # generated_ids_trimmed = generated_ids[:, input_token_len:] # Slice generated IDs
        # Correct way to handle batch dimension
        generated_ids_trimmed = [
             out_ids[input_token_len:] for out_ids in generated_ids
        ]


        # Decode the generated tokens
        response = tokenizer.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return response[0]  # Return the first (and only) decoded response


# --- extract_bbox_from_response is replaced by extract_all_bboxes_from_response ---
# def extract_bbox_from_response(response):
#    ... # Removed


# --- main function needs adjustment for output paths and merging ---
def main():
    parser = argparse.ArgumentParser(description="Visual Language Model Single Object Tracking Evaluation (One-Turn Prediction)")
    parser.add_argument("--model_path", type=str, default="/data1/lihaobo/LLaMA-Factory/saves/Qwen2.5-VL-3B-Instruct/full/tracking_large-2",
                        help="Model path")
    parser.add_argument("--dataset_name", type=str, default="OTB_lang",
                        help="Dataset name (e.g., OTB_lang, LaSOT, TrackingNet)")
    parser.add_argument("--output_dir", type=str, default="/data1/lihaobo/tracking/output_dir",
                        help="Base output directory for results and visualizations")
    parser.add_argument("--sequence", type=str, default=None, # Default to None to run all sequences
                        help="Specific sequence name to run (optional). If set, runs only this sequence on rank 0.")
    parser.add_argument("--max_new_tokens", type=int, default=2048, # Adjust based on sequence length and expected output format
                        help="Maximum new tokens to generate for the entire sequence prediction")
    # parser.add_argument("--context_window", type=int, default=2, # No longer relevant for one-turn prediction
    #                     help="Context window size for prediction")
    # parser.add_argument("--visualize", action="store_true", # Keep for compatibility? save_vis is better
    #                     help="Visualize tracking results (use --save_vis instead)")
    parser.add_argument("--save_vis", action="store_true", # Use action="store_true"
                        help="Save visualization results (images with predicted boxes)")
    # parser.add_argument("--vis_dir", type=str, default=None, # Output dir structure handles this now
    #                     help="Visualization results directory")
    parser.add_argument("--single_process", action="store_true",
                        help="Force single process execution even with multiple GPUs")

    args = parser.parse_args()

    # --- Setup specific output directory for this run ---
    run_output_dir = os.path.join(args.output_dir, args.dataset_name)
    if args.sequence:
        # If running a single sequence, results are directly in dataset dir, maybe add sequence name?
        # Let's keep it simple: results for single sequence go into the dataset dir.
        pass
        # run_output_dir = os.path.join(run_output_dir, args.sequence) # Optional: subdir for single sequence run

    # Create base results directory
    os.makedirs(run_output_dir, exist_ok=True)
    logger.info(f"Output will be saved in: {run_output_dir}")
    # Visualization directory will be inside run_output_dir/{sequence_name} if save_vis is True

    # Check if multi-processing can be used
    num_gpus = torch.cuda.device_count()
    use_multiprocessing = num_gpus > 1 and not args.single_process

    if use_multiprocessing:
        world_size = num_gpus
        logger.info(f"Detected {world_size} GPUs, enabling multi-process evaluation")
        # Ensure spawn method for CUDA safety
        if mp.get_start_method(allow_none=True) != 'spawn':
             mp.set_start_method('spawn', force=True)


        with Pool(world_size) as pool:
            # Use functools.partial to pass fixed arguments (world_size, args) to run_process
            func = functools.partial(run_process, world_size=world_size, args=args)
            # pool.map executes func for each rank in range(world_size)
            results_list = pool.map(func, range(world_size))

        # --- Merging results is no longer needed here ---
        # Individual processes now write directly to the final txt files in the output directory.
        # The `results_list` contains the structured results which might be useful for a final summary JSON,
        # but the primary output (TXT files for evaluation toolkits) is already generated.

        # Optional: Save a combined JSON summary if needed
        all_results_structure = []
        for result_sublist in results_list:
            if result_sublist: # Ensure sublist is not empty
                all_results_structure.extend(result_sublist)

        if all_results_structure:
             summary_json_path = os.path.join(args.output_dir, f"combined_results_{args.dataset_name}.json")
             try:
                 with open(summary_json_path, "w") as f:
                     json.dump(all_results_structure, f, indent=2)
                 logger.info(f"Multi-process evaluation complete. TXT results saved in {run_output_dir}. Combined JSON summary saved to {summary_json_path}")
             except Exception as e:
                  logger.error(f"Failed to save combined JSON summary: {e}")
        else:
             logger.info(f"Multi-process evaluation complete. TXT results saved in {run_output_dir}. No combined JSON summary generated (no results).")


    else: # Single process execution
        logger.info("Using single process evaluation")
        if args.sequence:
            logger.info(f"Evaluating single sequence: {args.sequence}")
        else:
            logger.info(f"Evaluating all sequences in dataset {args.dataset_name}")

        # Rank is 0, world_size is 1 for single process
        results_structure = run_process(rank=0, world_size=1, args=args)

        # Optional: Save JSON summary for single process run
        if results_structure:
            summary_json_path = os.path.join(args.output_dir, f"results_{args.dataset_name}")
            if args.sequence:
                summary_json_path += f"_{args.sequence}"
            summary_json_path += ".json"
            try:
                with open(summary_json_path, "w") as f:
                    json.dump(results_structure, f, indent=2)
                logger.info(f"Single process evaluation complete. TXT results saved in {run_output_dir}. JSON summary saved to {summary_json_path}")
            except Exception as e:
                 logger.error(f"Failed to save JSON summary: {e}")
        else:
             logger.info(f"Single process evaluation complete. TXT results saved in {run_output_dir}. No JSON summary generated (no results).")


if __name__ == "__main__":
    main()