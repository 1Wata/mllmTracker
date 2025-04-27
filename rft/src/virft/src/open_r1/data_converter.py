# --- Imports ---
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image
import torch
from dataclasses import dataclass
import os
import logging
from transformers import AutoProcessor # Or specific processor like LlavaProcessor, PaliGemmaProcessor

# --- Configuration ---
# # Set the path to your JSON dataset file (assuming JSON Lines format)
# json_dataset_path = "your_dataset.json" # <-- 重要：替换为你的JSON文件路径
# # Set the base path where your image folders (like 'OTB_sampled_mini4') are located
# image_base_path = "/path/to/your/image/directory" # <-- 重要：替换为你的图片所在的根目录
# # Specify the Hugging Face model/processor you intend to use
# # This determines how data is processed in the collator
# processor_name_or_path = "llava-hf/llava-1.5-7b-hf" # <-- 重要：替换为你实际使用的模型或processor


def create_dataset(data_path: str, image_base_path: str):
    """
    Create and process dataset for GRPO training with tracking format.
    
    Args:
        data_path: Path to the dataset JSON file
        image_base_path: Base path for image files
        
    Returns:
        Dataset with required fields: ['image_paths', 'problem', 'solution', 'prompt']
    """
    print("image_base_path:", image_base_path)
    
    # Load raw dataset
    raw_dataset = load_dataset('json', data_files=data_path, split='train')
    
    def process_data_for_tracking(example):
        """
        Process single example to tracking task format
        """
        if "messages" not in example or "images" not in example:
            return {}  # Return empty dict for filtering
            
        messages = example["messages"]
        relative_image_paths = example.get("images", [])
        
        if not messages or not relative_image_paths:
            return {}
            
        # Build full image paths
        full_image_paths = []
        for img_rel_path in relative_image_paths:
            full_img_path = os.path.join(image_base_path, img_rel_path)
            
            # Validate image file existence
            if not os.path.exists(full_img_path):
                print(f"Warning: Image path does not exist: {full_img_path}")
                continue
                
            full_image_paths.append(full_img_path)
            
        if not full_image_paths:
            return {}  # No valid images, return empty dict
            
        # Extract problem and solution
        problem = ""
        solution = ""
        
        for msg in messages:
            if msg["role"] == "user":
                # User message treated as problem
                if isinstance(msg["content"], str):
                    problem = msg["content"]
                elif isinstance(msg["content"], list):
                    # Handle multimodal content
                    problem = "".join([
                        item if isinstance(item, str) else "<image>"
                        for item in msg["content"]
                    ])
            elif msg["role"] == "assistant":
                # Assistant message treated as solution
                if isinstance(msg["content"], str):
                    solution = msg["content"]
                    # Add answer tags if not present
                    if "<answer>" not in solution:
                        # Try to extract bbox data and format it
                        import re
                        bbox_match = re.search(r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]', solution)
                        if bbox_match:
                            bbox = f"[{bbox_match.group(1)}, {bbox_match.group(2)}, {bbox_match.group(3)}, {bbox_match.group(4)}]"
                            solution = f"<answer>{bbox}</answer>"
                        else:
                            solution = f"<answer>{solution}</answer>"
        
        # Build prompt structure
        prompt = []
        for msg in messages:
            if msg["role"] in ["user", "system"]:
                # Add user and system messages
                new_msg = {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                prompt.append(new_msg)
                
        return {
            "image_paths": full_image_paths,
            "problem": problem,
            "solution": solution,
            "prompt": prompt
        }
    
    # Apply processing function and remove original columns
    processed_dataset = raw_dataset.map(
        process_data_for_tracking,
        remove_columns=raw_dataset.column_names
    )
    
    # Filter out empty samples
    original_count = len(processed_dataset)
    final_dataset = processed_dataset.filter(
        lambda example: bool(example.get("prompt")) and bool(example.get("solution"))
    )
    
    print(f"Original samples: {original_count}, processed samples: {len(final_dataset)}")
    
    # Create train/test split
    if len(final_dataset) > 10:
        train_size = int(0.9 * len(final_dataset))
        test_size = len(final_dataset) - train_size
        
        dataset_dict = DatasetDict({
            'train': final_dataset.select(range(train_size)),
            'test': final_dataset.select(range(train_size, len(final_dataset)))
        })
    else:
        # Too few samples, use all for training
        dataset_dict = DatasetDict({
            'train': final_dataset,
            'test': final_dataset.select(range(min(len(final_dataset), 2)))  # Use at least 2 samples for testing
        })
    
    print(f"Train set size: {len(dataset_dict['train'])}, test set size: {len(dataset_dict['test'])}")
    return dataset_dict

def load_image(examples):
    """
    Load images and add them to samples
    
    Args:
        examples: Batch of samples containing image_paths
        
    Returns:
        Samples with loaded images
    """
    images = []
    for path in examples['image_paths']:
        image = Image.open(path).convert("RGB")
        images.append(image)

    
    examples['image'] = images
    return examples


@dataclass
class MultimodalDataCollator:
    processor: object # Expects an instance of a Hugging Face processor

    def __call__(self, features):
        """
        Collates a list of features into a batch. Loads images from paths.
        Args:
            features (list[dict]): A list of dictionaries, each processed by process_sharegpt_for_paths.
                                   Expected keys: 'prompt' (list of messages), 'image_paths' (list of str).
        Returns:
            dict: A batch dictionary suitable for the model, typically containing 'input_ids',
                  'attention_mask', 'pixel_values', and potentially 'labels'.
        """
        batch = {"image_paths": [], "pil_images": [], "prompts": []}
        for feature in features:
            batch["prompts"].append(feature["prompt"])
            batch["image_paths"].append(feature.get("image_paths", [])) # list of paths for this sample

        # Load images using PIL
        loaded_images_batch = [] # List of lists (or list of None if no images for a sample)
        for paths_for_sample in batch["image_paths"]:
            images_for_sample = []
            if paths_for_sample:
                for path in paths_for_sample:
                    try:
                        img = Image.open(path).convert("RGB")
                        images_for_sample.append(img)
                    except FileNotFoundError:
                        # CRITICAL: How to handle? Skipping might break models.
                        # Options: Append None, append a placeholder image, skip the whole sample?
                        # Appending None might work if processor handles it. Let's try appending None.
                        images_for_sample.append(None) # Append None placeholder
                    except Exception as e:
                        images_for_sample.append(None) # Append None placeholder
                # Filter out None values *if* the processor cannot handle them
                images_for_sample = [img for img in images_for_sample if img is not None]
                if not images_for_sample and paths_for_sample:
                     # If NO images loaded but some were expected, append None or handle as per model needs
                     loaded_images_batch.append(None) # Indicate failure for this sample's images
                else:
                    loaded_images_batch.append(images_for_sample) # Append list of loaded images
            else:
                loaded_images_batch.append(None) # No images were specified for this sample

        # --- Processor-Specific Part ---
        # How you format text and pass images HIGHLY DEPENDS on the processor.
        # Example for Llava-like processor (needs careful adaptation):
        # It might expect a flat list of PIL images and text prompts derived from messages.
        # It often uses special tokens like <image> within the text.
        # You might need to format the 'prompt' (list of messages) into the correct
        # conversational format string expected by the processor/tokenizer *before* calling it.

        # Placeholder: Prepare text inputs based on processor needs
        # This likely involves iterating through batch['prompts'] and formatting them
        # correctly, potentially inserting <image> tokens based on loaded_images_batch.
        # This logic is complex and processor-dependent.
        formatted_text_inputs = []
        images_to_process = []

        for i, prompt_messages in enumerate(batch["prompts"]):
            # --- !! This section needs careful implementation based on your chosen model/processor !! ---
            # Example: Concatenate messages, maybe add role info, insert <image> tokens
            # matching the number of successfully loaded images in loaded_images_batch[i]
            conversation = ""
            image_token = self.processor.tokenizer.decode(self.processor.image_token_index) # Get the actual image token string e.g '<image>'
            num_images_for_sample = len(loaded_images_batch[i]) if loaded_images_batch[i] is not None else 0
            image_counter = 0
            for msg in prompt_messages:
                role = msg.get("role", "user") # Default to user? Adjust as needed
                content = msg.get("content", "")
                # Naive: Add image token(s) before user message if images exist for this sample?
                # More sophisticated: Detect placeholders like `<image>filename.jpg` in content
                # and insert image_token accordingly.
                # This example assumes images generally precede the user message asking about them.
                if role == "user" and image_counter < num_images_for_sample:
                    # Add one image token per image associated with this turn (simplistic)
                    # This mapping is often ambiguous in ShareGPT format!
                    # A better format would explicitly link images to messages.
                    # Assuming one image per user turn for simplicity here:
                    # conversation += f"{image_token}\n" # Add image token
                    # images_to_process.append(loaded_images_batch[i][image_counter]) # Add the corresponding PIL image
                    # image_counter += 1
                    pass # Placeholder - requires specific logic

                # Format the message (adjust as per model requirements)
                conversation += f"{role.upper()}: {content}\n" # Example formatting

            # Add image tokens and corresponding images based on your convention
            # For this example, let's assume all images for a sample are processed together
            # with the full conversation text. This matches some models like Llava.
            num_images_in_text = conversation.count(image_token) # If you inserted tokens above
            # Or assume tokens are already in the original message content:
            # You'd need to parse content to find tokens/placeholders

            # Simplified: Pass all loaded images for the sample to the processor along with text
            formatted_text_inputs.append(conversation)
            if loaded_images_batch[i]:
                images_to_process.extend(loaded_images_batch[i]) # Add all images for this sample


        # ---- Processor Call ----
        # The actual arguments depend on the processor class. Check its documentation.

        inputs = self.processor(
            text=formatted_text_inputs,       # The formatted conversation strings
            images=images_to_process if images_to_process else None, # List of PIL Images or None
            return_tensors="pt",
            padding=True,                 # Pad text inputs
            truncation=True               # Truncate text inputs if needed
            # Add other processor-specific arguments, e.g., max_length
        )

        # The processor should return a dictionary containing 'input_ids', 'attention_mask', 'pixel_values', etc.
        # If you need 'labels' for training (e.g., predicting the assistant's response),
        # the processor or custom logic here needs to handle shifting input_ids appropriately.
        # This is often handled by the processor itself or standard data collators like DataCollatorForSeq2Seq.
        # You might need to adapt this collator or combine it with another if labels are required.



# Instantiate the custom data collator
# data_collator = MultimodalDataCollator(processor=processor)

# Now you can pass `final_dataset` and `data_collator` to the Trainer:
# trainer = Trainer(
#     model=your_model,               # Your loaded multi-modal model
#     args=training_args,             # Your TrainingArguments
#     train_dataset=final_dataset,
#     # eval_dataset=processed_eval_dataset, # Process eval dataset similarly
#     tokenizer=processor.tokenizer,  # Often the processor contains the tokenizer
#     data_collator=data_collator,
#     ...
# )

# Example: Test the data collator with a small sample
# from torch.utils.data import DataLoader
# logger.info("Testing data collator with a small batch...")
# try:
#     sample_batch = [final_dataset[i] for i in range(min(4, len(final_dataset)))] # Get a small batch
#     batch_output = data_collator(sample_batch)
#     logger.info("Data collator test output keys:")
#     print(batch_output.keys())
#     logger.info("Data collator test successful.")
#     # print("Example batch output:")
#     # print(batch_output) # Inspect the structure
# except Exception as e:
#      logger.error(f"Error testing data collator: {e}")