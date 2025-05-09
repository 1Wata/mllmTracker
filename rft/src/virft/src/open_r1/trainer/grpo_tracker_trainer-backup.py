# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union
from PIL import Image
import inspect
from transformers.image_processing_utils import get_image_size

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import copy
import math
from open_r1.utils.utils import transform_bbox, smart_resize

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]




class Qwen25VLGRPOTrainer(Trainer):
    """
    Trainer for Qwen2.5VL model using Group Relative Policy Optimization (GRPO) method for single object tracking tasks.

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either a model ID string or a pretrained model object.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions for computing rewards. Can be a single function or a list of functions.
        args (`GRPOConfig`, *optional*, defaults to `None`):
            Configuration for the trainer.
        train_dataset (`~datasets.Dataset` or `~datasets.IterableDataset`):
            Dataset to use for training. Must include a "prompt" column.
        eval_dataset (`~datasets.Dataset`, `~datasets.IterableDataset` or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation.
        processing_class (`~transformers.PreTrainedTokenizerBase`, *optional*, defaults to `None`):
            Processing class used to process the data.
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            Tuple containing the optimizer and scheduler.
        peft_config (`~peft.PeftConfig`, *optional*, defaults to `None`):
            PEFT configuration for model fine-tuning.
        max_pixels (`int`, *optional*, defaults to 12845056):
            Maximum number of pixels for image processing.
        min_pixels (`int`, *optional*, defaults to 3136):
            Minimum number of pixels for image processing.
        attn_implementation (`str`, defaults to "flash_attention_2"):
            Implementation method for attention mechanism.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056, # Default from Qwen, adjust as needed
        min_pixels: Optional[int] = 3136,    # Default from Qwen, adjust as needed
        attn_implementation: str = "flash_attention_2",
    ):
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        # Configure args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Prepare model initialization
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        
        if isinstance(model, str):
            model_id = model
            # Handle torch dtype
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, str) and torch_dtype != "auto":
                model_init_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
            elif not (isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None):
                 raise ValueError(f"Invalid torch_dtype: {torch_dtype}")
            model_init_kwargs["use_cache"] = False if args.gradient_checkpointing else model_init_kwargs.get("use_cache", True)
            
            # Ensure torch_dtype is set, defaulting to bfloat16 for Qwen models if not specified
            if model_init_kwargs.get("torch_dtype") is None and ("Qwen2.5-VL" in model_id or "Qwen2-VL" in model_id) :
                model_init_kwargs["torch_dtype"] = torch.bfloat16
            
            if "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs:
                print("Warning: `model_init_kwargs` was passed but model is already instantiated. It will be ignored.")

        # Apply PEFT if needed
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Setup reference model
        if peft_config is None: # Standard full model fine-tuning
            if is_deepspeed_zero3_enabled(): # DeepSpeed Zero3 requires loading ref model from scratch
                ref_model_load_kwargs = model_init_kwargs.copy() # Use same kwargs for ref model
                if "Qwen2.5-VL" in model_id:
                    self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **ref_model_load_kwargs)
                elif "Qwen2-VL" in model_id:
                    self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **ref_model_load_kwargs)
                else:
                    self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **ref_model_load_kwargs)
            else:
                # Create reference model from initial model
                self.ref_model = create_reference_model(model)
        else: # PEFT fine-tuning
            self.ref_model = None # Adapters will be disabled on the main model to get reference behavior

        # Setup processing class
        if processing_class is None:
            if "Qwen2.5-VL" in model_id or "Qwen2-VL" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                # Ensure pad_token_id is set on the processor itself if tokenizer has it
                if hasattr(processing_class, 'tokenizer') and processing_class.tokenizer.pad_token_id is not None:
                    processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
                else: # Fallback: use eos_token_id if pad_token_id is None
                    processing_class.pad_token_id = processing_class.tokenizer.eos_token_id
                
                if hasattr(processing_class, 'tokenizer') and processing_class.tokenizer.eos_token_id is not None:
                     processing_class.eos_token_id = processing_class.tokenizer.eos_token_id

                if hasattr(processing_class, 'image_processor'):
                    setattr(processing_class.image_processor, 'max_pixels', max_pixels)
                    setattr(processing_class.image_processor, 'min_pixels', min_pixels)
            else: # Fallback for non-VL models or if specific processor logic is not needed
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
        
        # Ensure pad_token_id is set on processing_class, fallback to eos_token_id if necessary
        if not hasattr(processing_class, 'pad_token_id') or processing_class.pad_token_id is None:
            if hasattr(processing_class, 'tokenizer') and processing_class.tokenizer.pad_token_id is not None:
                processing_class.pad_token_id = processing_class.tokenizer.pad_token_id
            elif hasattr(processing_class, 'eos_token_id') and processing_class.eos_token_id is not None:
                print("Warning: pad_token_id not set on processing_class, using eos_token_id as pad_token_id.")
                processing_class.pad_token_id = processing_class.eos_token_id
            elif hasattr(processing_class, 'tokenizer') and processing_class.tokenizer.eos_token_id is not None:
                print("Warning: pad_token_id not set on processing_class or its tokenizer, using tokenizer.eos_token_id as pad_token_id.")
                processing_class.pad_token_id = processing_class.tokenizer.eos_token_id
            else:
                raise ValueError("Cannot determine pad_token_id for the processing_class.")

        if not hasattr(processing_class, 'eos_token_id') or processing_class.eos_token_id is None:
            if hasattr(processing_class, 'tokenizer') and processing_class.tokenizer.eos_token_id is not None:
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            else:
                # This might be problematic for generation if EOS is not clearly defined.
                print("Warning: eos_token_id not found on processing_class or its tokenizer.")


        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_model_kwargs = model_init_kwargs.copy() # Use similar kwargs for reward model
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **reward_model_kwargs
                )
        self.reward_funcs = reward_funcs

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        if len(reward_processing_classes) != len(reward_funcs):
            raise ValueError("Length of reward_processing_classes must match reward_funcs.")

        for i, (rew_proc_cls, rew_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(rew_func, PreTrainedModel):
                if rew_proc_cls is None:
                    rew_proc_cls = AutoTokenizer.from_pretrained(rew_func.config._name_or_path)
                if not hasattr(rew_proc_cls, 'pad_token_id') or rew_proc_cls.pad_token_id is None:
                    rew_proc_cls.pad_token_id = rew_proc_cls.eos_token_id # Common practice
                rew_func.config.pad_token_id = rew_proc_cls.pad_token_id
                reward_processing_classes[i] = rew_proc_cls
        self.reward_processing_classes = reward_processing_classes

        # Define simple data collator
        def data_collator(features):
            return features

        # Configure training parameters
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=args.temperature, # Use temperature from GRPOConfig
            num_return_sequences=self.num_generations,
            pad_token_id=processing_class.pad_token_id,
            eos_token_id=processing_class.eos_token_id
        )
        self.beta = args.beta

        # Suppress warnings about token estimation
        model.warnings_issued["estimate_tokens"] = True

        # Initialize metrics storage
        self._metrics = defaultdict(list)

        # Initialize trainer
        super().__init__(
            model=model,
            args=args,
            # data_collator=custom_tracking_data_collator,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Configure loss handling for gradient accumulation
        self.model_accepts_loss_kwargs = False

        # Prepare reference model and reward functions
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        """Set signature columns needed for data preprocessing"""
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "problem", "solution"] # Add all expected keys from dataset

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        """
        Calculate per-token log probabilities for the given model and inputs
        
        Args:
            model: The model to use for prediction
            input_ids: Token IDs
            attention_mask: Attention mask
            pixel_values: Image pixel values
            image_grid_thw: Image grid information
            
        Returns:
            Tensor of per-token log probabilities
        """
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values, 
            image_grid_thw=image_grid_thw
        )
        logits = outputs.logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V)
        input_ids = input_ids[:, 1:]  # (B, L-1)
        
        # Calculate log probabilities one row at a time to save memory
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
            
        return torch.stack(per_token_logps)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """Ensure inputs are moved to the correct device."""
        prepared_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared_inputs[k] = v.to(self.accelerator.device)
            else:
                prepared_inputs[k] = v # Non-tensor data (like list of PIL images)
        return prepared_inputs

    def _calculate_image_sizes(self, images_per_sample):
        """
        计算每个样本的原始尺寸和调整后的尺寸，使用smart_resize确保符合模型要求
        
        Args:
            images_per_sample: 列表的列表，每个内部列表包含一个样本的所有图像
            
        Returns:
            tuple: (original_sizes, resized_sizes) 包含每个样本的尺寸信息
        """
        original_sizes = []
        resized_sizes = []
        
        image_processor = getattr(self.processing_class, 'image_processor', None)
        if image_processor is None:
            print("Warning: self.processing_class.image_processor is not available. Using default/fallback size parameters for _calculate_image_sizes.")
            min_pixels_ip = self.min_pixels
            max_pixels_ip = self.max_pixels
            # Fallback factor if image_processor or its attributes are not available
            factor = 28 # A common default (e.g., 14 * 2 for patch_size * merge_size)
        else:
            min_pixels_ip = getattr(image_processor, 'min_pixels', self.min_pixels)
            max_pixels_ip = getattr(image_processor, 'max_pixels', self.max_pixels)
            if hasattr(image_processor, 'size_divisor'): # Common for ViTFeatureExtractor
                factor = image_processor.size_divisor
            elif hasattr(image_processor, 'patch_size'): # Qwen specific
                patch_size = image_processor.patch_size
                merge_size = getattr(image_processor, 'merge_size', 2) 
                factor = patch_size * merge_size
            else: 
                factor = 28 
                print(f"Warning: Could not determine 'factor' from image_processor. Using default: {factor}")

        # Qwen系列模型的默认参数
        # patch_size = 14  # 默认patch大小
        # merge_size = 2   # 默认merge大小
        # factor = patch_size * merge_size  # 缩放因子
        
        # 遍历批次中的每个样本
        for image_list_for_sample in images_per_sample:
            if not image_list_for_sample or not isinstance(image_list_for_sample[0], Image.Image):
                # For academic code, we might assume valid inputs or let it fail.
                # If strict, raise error. If lenient, append placeholders and warn.
                raise ValueError("Encountered a sample with an empty or invalid image list.")
                # original_sizes.append(None) 
                # resized_sizes.append(None)
                # continue
            
            # 使用第一张图像确定该样本的分辨率（假设同一样本内所有图像分辨率相同）
            first_image = image_list_for_sample[0]
            original_width, original_height = first_image.size
            # Store as (width, height)
            original_sizes.append((original_width, original_height))
            
            # 使用smart_resize计算调整后的尺寸
            try:
                resized_height, resized_width = smart_resize(
                    original_height, 
                    original_width,
                    factor=factor,
                    min_pixels=min_pixels_ip,
                    max_pixels=max_pixels_ip
                )
                # Store as (width, height)
                resized_sizes.append((resized_width, resized_height))
            except ValueError as e:
                # 处理特殊情况，例如过小的图像或极端宽高比
                print(f"Warning: {e}. Using fallback resizing method for image size ({original_height}, {original_width}).")
                # 使用原始简单缩放作为备选方案
                num_pixels = original_height * original_width
                scale = 1.0
                if num_pixels > max_pixels_ip:
                    scale = (max_pixels_ip / num_pixels) ** 0.5
                elif num_pixels < min_pixels_ip:
                    scale = (min_pixels_ip / num_pixels) ** 0.5
                
                # 确保尺寸可被factor整除
                resized_height = int(original_height * scale + 0.5)
                resized_height = math.ceil(resized_height / factor) * factor
                
                resized_width = int(original_width * scale + 0.5)
                resized_width = math.ceil(resized_width / factor) * factor
                
                # Store as (width, height)
                resized_sizes.append((resized_width, resized_height))
        
        return original_sizes, resized_sizes

    def _generate_completions_and_masks(self, model, prompt_inputs_on_device: dict[str, torch.Tensor]):
        prompt_ids = prompt_inputs_on_device["input_ids"]
        prompt_mask = prompt_inputs_on_device["attention_mask"]
        pixel_values_for_generation = prompt_inputs_on_device["pixel_values"]
        image_grid_thw_for_generation = prompt_inputs_on_device["image_grid_thw"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        generation_inputs = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "pixel_values": pixel_values_for_generation,
            "image_grid_thw": image_grid_thw_for_generation,
        }
        
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **generation_inputs, 
                generation_config=self.generation_config
            )

        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]
        prompt_mask_repeated = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        is_eos = completion_ids == self.processing_class.eos_token_id
        device = completion_ids.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        any_eos_along_dim1 = is_eos.any(dim=1)
        if any_eos_along_dim1.any(): 
            eos_idx[any_eos_along_dim1] = is_eos.int().argmax(dim=1)[any_eos_along_dim1]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask_repeated = torch.cat([prompt_mask_repeated, completion_mask], dim=1)
        pixel_values_final_repeated = pixel_values_for_generation.repeat_interleave(self.num_generations, dim=0)
        image_grid_thw_final_repeated = image_grid_thw_for_generation.repeat_interleave(self.num_generations, dim=0)
        
        return {
            "prompt_completion_ids": prompt_completion_ids,
            "completion_ids": completion_ids,
            "attention_mask_repeated": attention_mask_repeated,
            "pixel_values_final_repeated": pixel_values_final_repeated,
            "image_grid_thw_final_repeated": image_grid_thw_final_repeated,
            "prompt_length": prompt_length,
            "completion_mask": completion_mask,
        }

    def _calculate_logprobs_and_kl(self, model, ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw, prompt_length):
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        with torch.inference_mode():
            if ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                )
            else: # PEFT case: disable adapter on the main model
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                    )
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        return per_token_logps, ref_per_token_logps, per_token_kl

    def _calculate_advantages_and_final_loss(self, rewards, per_token_logps, old_per_token_logps, per_token_kl, completion_mask):
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        loss_term = torch.exp(per_token_logps - old_per_token_logps) * advantages.unsqueeze(1)
        per_token_loss = -(loss_term - self.beta * per_token_kl)
            
        loss = ((per_token_loss * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)).mean()
        return loss, advantages, std_grouped_rewards

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPO Trainer does not support returning outputs")

        prompts_orig_text_for_reward = [x["prompt"] for x in inputs] # Keep original prompts for reward
        system_prompts = [x.get("problem", "") for x in inputs]

        formatted_examples = []
        for i, example_data in enumerate(inputs):
            formatted_example = copy.deepcopy(example_data)
            if system_prompts[i]:
                if is_conversational(example_data):
                    if not any(msg.get("role") == "system" for msg in formatted_example["prompt"]):
                        formatted_example["prompt"].insert(0, {"role": "system", "content": system_prompts[i]})
                else: # Non-conversational, add system_prompt key for maybe_apply_chat_template
                    formatted_example["system_prompt"] = system_prompts[i]
            formatted_examples.append(formatted_example)

        prompts_text_templated = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in formatted_examples]
        images_per_sample = [x["image"] for x in inputs] # List of lists of PIL Images
        
        # Assuming all inputs are valid and _calculate_image_sizes is robust
        original_sizes, resized_sizes = self._calculate_image_sizes(images_per_sample)

        # Transform bboxes in prompts
        for i, prompt_text_item in enumerate(prompts_text_templated):
            original_size_item = original_sizes[i]
            resized_size_item = resized_sizes[i]
            # If original_size_item or resized_size_item is None (e.g. from a lenient _calculate_image_sizes)
            # this would error. For academic code, we assume valid sizes or let it fail.
            if original_size_item is None or resized_size_item is None:
                 raise ValueError(f"Sample {i} has invalid image sizes after _calculate_image_sizes.")

            import re
            bbox_pattern = r"bounding box for template frame .+ is: \[(\d+), (\d+), (\d+), (\d+)\]"
            matches = re.findall(bbox_pattern, prompt_text_item)
            temp_prompt_text = prompt_text_item
            for match_coords in matches:
                x1, y1, x2, y2 = map(int, match_coords)
                original_bbox = [x1, y1, x2, y2]
                if original_size_item[0] == 0 or original_size_item[1] == 0 or \
                   resized_size_item[0] == 0 or resized_size_item[1] == 0:
                    print(f"Warning: Skipping bbox transformation for '{match_coords}' due to zero image dimension for sample {i}.")
                    continue # Skip this bbox transformation
                resized_bbox = transform_bbox(original_bbox, original_size_item, resized_size_item, 'original_to_resized')
                if resized_bbox:
                    new_bbox_str = f"[{int(resized_bbox[0])}, {int(resized_bbox[1])}, {int(resized_bbox[2])}, {int(resized_bbox[3])}]"
                    old_bbox_str = f"[{x1}, {y1}, {x2}, {y2}]"
                    temp_prompt_text = temp_prompt_text.replace(old_bbox_str, new_bbox_str)
            prompts_text_templated[i] = temp_prompt_text
        
        # Tokenize and prepare model inputs
        processor_kwargs = {
            "text": prompts_text_templated,
            "images": images_per_sample, # Pass list of lists of PIL Images
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left", 
            "add_special_tokens": False, 
        }
        prompt_inputs_from_tokenizer = self.processing_class(**processor_kwargs)
        prompt_inputs_on_device = super()._prepare_inputs(prompt_inputs_from_tokenizer)

        # Generate completions and masks
        generated_data_dict = self._generate_completions_and_masks(model, prompt_inputs_on_device)
        
        prompt_completion_ids = generated_data_dict["prompt_completion_ids"]
        attention_mask_repeated = generated_data_dict["attention_mask_repeated"]
        pixel_values_final_repeated = generated_data_dict["pixel_values_final_repeated"]
        image_grid_thw_final_repeated = generated_data_dict["image_grid_thw_final_repeated"]
        prompt_length = generated_data_dict["prompt_length"]
        completion_mask = generated_data_dict["completion_mask"]

        # Calculate log probabilities and KL divergence
        current_per_token_logps, _, per_token_kl = \
            self._calculate_logprobs_and_kl(model, self.ref_model, prompt_completion_ids, attention_mask_repeated,
                                            pixel_values_final_repeated, image_grid_thw_final_repeated, prompt_length)
        
        old_per_token_logps_detached = current_per_token_logps.detach()
        
        # Decode completions
        completions_decoded = self.processing_class.batch_decode(generated_data_dict["completion_ids"], skip_special_tokens=True)
        
        is_conv_sample = is_conversational(inputs[0]) # Assume structure of first original input is representative

        if is_conv_sample:
            completions_for_reward = [[{"role": "assistant", "content": c}] for c in completions_decoded]
        else:
            completions_for_reward = completions_decoded

        # Prepare for reward calculation
        # Use original, untemplated prompts for reward calculation, repeated G times
        prompts_for_reward_calc = [p_orig for p_orig in prompts_orig_text_for_reward for _ in range(self.num_generations)]
        
        reward_kwargs = {}
        # Get keys from the first input, excluding standard ones
        keys_for_reward_kwargs = [
            key for key in inputs[0].keys() 
            if key not in ["prompt", "completion", "image", "problem", "system_prompt"] # "system_prompt" was temporary
        ]
        for key_r in keys_for_reward_kwargs:
            reward_kwargs[key_r] = []
            for original_input_sample in inputs: # Iterate over original batch
                reward_kwargs[key_r].extend([original_input_sample.get(key_r)] * self.num_generations)

        reward_kwargs['original_size'] = [size for size in original_sizes for _ in range(self.num_generations)]
        reward_kwargs['resized_size'] = [size for size in resized_sizes for _ in range(self.num_generations)]

        # Compute rewards
        rewards, rewards_per_func = self._compute_rewards_from_functions(
            inputs, # Pass original inputs for context if reward functions need it
            prompts_for_reward_calc, 
            completions_for_reward, 
            reward_kwargs
        )

        # Calculate advantages and final loss
        loss, _, std_grouped_rewards = \
            self._calculate_advantages_and_final_loss(rewards, current_per_token_logps, 
                                                      old_per_token_logps_detached, 
                                                      per_token_kl, completion_mask)

        # Log metrics
        self._metrics["completion_length"].append(self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item())
        gathered_rewards_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i_rf, reward_func_item in enumerate(self.reward_funcs):
            if isinstance(reward_func_item, PreTrainedModel):
                reward_func_name = reward_func_item.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func_item.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(gathered_rewards_per_func[i_rf].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        
        # Calculate mean of std dev per group for reward_std
        # std_grouped_rewards is (B*G), reshape to (B,G), take std over G, then mean over B
        mean_of_group_stds = std_grouped_rewards.view(-1, self.num_generations).std(dim=1).mean()
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(mean_of_group_stds).mean().item())

        if self.beta > 0:
            mean_kl_val = ((per_token_kl * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl_val).mean().item())
        
        return loss

    def _compute_rewards_from_functions(self, original_inputs_batch, prompts, completions, reward_kwargs):
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        # Determine conversational nature from the first original input
        is_conv_sample = is_conversational(original_inputs_batch[0]) if original_inputs_batch else False

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conv_sample:
                    # prompts are original prompts (text) repeated G times
                    # completions are list of dicts [{"role": "assistant", "content": c}] repeated G times
                    # We need to combine them correctly.
                    # The original prompt text needs to be converted to conversational format if not already.
                    # For simplicity, assume prompts are already in the correct format (text or list of dicts)
                    # and completions are also correctly formatted.
                    # The `prompts` here are original text prompts, repeated.
                    # `completions` are list of [{"role": "assistant", "content": c}]
                    
                    # This part needs to be careful: `prompts` are List[str] (original prompt text repeated G times)
                    # `completions` are List[List[Dict]] (assistant's turn, repeated G times)
                    # We need to construct the full conversation for each.
                    
                    messages_for_reward = []
                    for idx_orig_input in range(len(original_inputs_batch)):
                        original_prompt_structure = original_inputs_batch[idx_orig_input]["prompt"] # This is List[Dict]
                        for g in range(self.num_generations):
                            current_completion_turn = completions[idx_orig_input * self.num_generations + g] # This is List[Dict]
                            # Ensure original_prompt_structure is a list of dicts
                            if isinstance(original_prompt_structure, str): # Should not happen if is_conv_sample is true
                                 full_convo = [{"role":"user", "content": original_prompt_structure}] + current_completion_turn
                            else:
                                 full_convo = original_prompt_structure + current_completion_turn
                            messages_for_reward.append({"messages": full_convo})
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages_for_reward]

                else: # Non-conversational
                    # prompts are List[str] (original prompt text repeated G times)
                    # completions are List[str] (assistant's response text repeated G times)
                    texts = [p_text + c_text for p_text, c_text in zip(prompts, completions)]

                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = {k: v.to(next(reward_func.parameters()).device) for k, v in reward_inputs.items()}
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else: # Custom reward function
                # Custom reward functions might need specific keys from reward_kwargs
                # Ensure 'solution' is handled correctly if present in reward_kwargs
                # All kwargs should be (B*G) length
                current_reward_kwargs_for_custom_func = reward_kwargs.copy() 
                solutions_for_custom_func = current_reward_kwargs_for_custom_func.pop("solution", [None] * len(prompts))
                
                output_reward_func = reward_func(
                    completions=completions, # completions are already (B*G)
                    solution=solutions_for_custom_func, 
                    **current_reward_kwargs_for_custom_func 
                )
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
        
        rewards = rewards_per_func.sum(dim=1)
        return rewards, rewards_per_func

    # ... (log method and other helper methods like _set_signature_columns_if_needed, _get_per_token_logps, _prepare_inputs)
    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Log training metrics"""
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items() if val} # average the metrics, handle empty val
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = ["prompt", "image", "problem", "solution"] # Add all expected keys from dataset

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """Ensure inputs are moved to the correct device."""
        prepared_inputs = {}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                prepared_inputs[k] = v.to(self.accelerator.device)
            else:
                prepared_inputs[k] = v # Non-tensor data (like list of PIL images)
        return prepared_inputs

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        outputs = model(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            pixel_values=pixel_values, 
            image_grid_thw=image_grid_thw
        )
        logits = outputs.logits
        # Shift logits and labels for next token prediction
        # Logits: (B, L, V), Labels: (B, L)
        # We want log P(token_i | token_<i, image)
        # So, for logits at position j (predicting token j+1), we need label at position j+1
        shifted_logits = logits[:, :-1, :] # (B, L-1, V)
        shifted_labels = input_ids[:, 1:]   # (B, L-1)
        
        log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)
        
        # Gather the log probabilities of the true tokens
        # shifted_labels.unsqueeze(-1) makes it (B, L-1, 1) to be used with gather
        per_token_logps = torch.gather(log_probs, dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1) # (B, L-1)
            
        return per_token_logps



