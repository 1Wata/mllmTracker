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
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
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
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
                
            # Disable caching for gradient checkpointing
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            

            model_init_kwargs["torch_dtype"] = torch.bfloat16
            # exit(0)
            # Load appropriate model type based on model_id
            if "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        # Apply PEFT if needed
        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Setup reference model
        if peft_config is None:
            if is_deepspeed_zero3_enabled():
                if "Qwen2.5-VL" in model_id:
                    self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                elif "Qwen2-VL" in model_id:
                    self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
                else:
                    self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
            else:
                # Create reference model from initial model
                self.ref_model = create_reference_model(model)
        else:
            # No need for reference model with PEFT
            self.ref_model = None

        # Setup processing class
        if processing_class is None:
            if "Qwen2.5-VL" in model_id or "Qwen2-VL" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                # Set image processor parameters
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Setup reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
            
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Setup reward processing classes
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # Set pad token ID for reward model
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
                
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
            temperature=1.0,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
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
            self._signature_columns = ["prompt"]

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
            input_ids, 
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
        """Custom input preparation for GRPO trainer"""
        return inputs

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
        image_processor = self.processing_class.image_processor
        min_pixels = getattr(image_processor, 'min_pixels', self.min_pixels)
        max_pixels = getattr(image_processor, 'max_pixels', self.max_pixels)
        
        # Qwen系列模型的默认参数
        patch_size = 14  # 默认patch大小
        merge_size = 2   # 默认merge大小
        factor = patch_size * merge_size  # 缩放因子
        
        # 遍历批次中的每个样本
        for image_list_for_sample in images_per_sample:
            if not image_list_for_sample:
                raise ValueError("Encountered a sample with an empty image list.")
            
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
                    min_pixels=min_pixels,
                    max_pixels=max_pixels
                )
                # Store as (width, height)
                resized_sizes.append((resized_width, resized_height))
            except ValueError as e:
                # 处理特殊情况，例如过小的图像或极端宽高比
                print(f"Warning: {e}. Using fallback resizing method for image size ({original_height}, {original_width}).")
                # 使用原始简单缩放作为备选方案
                num_pixels = original_height * original_width
                scale = 1.0
                if num_pixels > max_pixels:
                    scale = (max_pixels / num_pixels) ** 0.5
                elif num_pixels < min_pixels:
                    scale = (min_pixels / num_pixels) ** 0.5
                
                # 确保尺寸可被factor整除
                resized_height = int(original_height * scale + 0.5)
                resized_height = math.ceil(resized_height / factor) * factor
                
                resized_width = int(original_width * scale + 0.5)
                resized_width = math.ceil(resized_width / factor) * factor
                
                # Store as (width, height)
                resized_sizes.append((resized_width, resized_height))
        
        return original_sizes, resized_sizes

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute GRPO loss for tracking task

        Args:
            model: The model being trained
            inputs: Input data dictionary (list of dicts)
            return_outputs: Whether to return outputs (not supported)
            num_items_in_batch: Number of items in batch

        Returns:
            The computed loss value
        """
        if return_outputs:
            raise ValueError("The GRPO Trainer does not support returning outputs")


        prompts = [x["prompt"] for x in inputs]
        # 获取system prompt (problem字段)
        system_prompts = [x.get("problem", "") for x in inputs]  # 使用get安全地获取problem字段

        # 创建包含system prompt的完整输入
        formatted_examples = []
        for i, example in enumerate(inputs):
            # 深拷贝避免修改原始输入
            formatted_example = copy.deepcopy(example)
            # 如果存在system prompt，将其添加到适当位置
            if system_prompts[i]:
                # 如果数据是对话格式，确保system prompt作为对话的第一部分
                if is_conversational(example):
                    if not any(msg.get("role") == "system" for msg in formatted_example["prompt"]):
                        formatted_example["prompt"].insert(0, {"role": "system", "content": system_prompts[i]})
                else:
                    formatted_example["system_prompt"] = system_prompts[i]
            formatted_examples.append(formatted_example)

        # 使用处理后的examples
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in formatted_examples]
        images_per_sample = [x["image"] for x in inputs]

        # --- Start: Flatten images and get counts ---
        flat_images = [img for sample_images in images_per_sample for img in sample_images]
        image_counts = [len(sample_images) for sample_images in images_per_sample]
        # --- End: Flatten images and get counts ---

        # original_sizes and resized_sizes are now (W, H)
        original_sizes, resized_sizes = self._calculate_image_sizes(images_per_sample)


        for i, prompt_text in enumerate(prompts_text):
            original_size = original_sizes[i] # (W, H)
            resized_size = resized_sizes[i]   # (W, H)
            # 使用正则表达式查找并替换bbox文本
            import re
            bbox_pattern = r"bounding box for template frame .+ is: \[(\d+), (\d+), (\d+), (\d+)\]"
            matches = re.findall(bbox_pattern, prompt_text)
            
            for match in matches:
                x1, y1, x2, y2 = map(int, match)
                original_bbox = [x1, y1, x2, y2]
                # transform_bbox from utils expects (W, H), which is now provided correctly
                resized_bbox = transform_bbox(original_bbox, original_size, resized_size, 'original_to_resized')
                # new_to_old = transform_bbox(resized_bbox, original_size, resized_size, 'resized_to_original')
                if resized_bbox:
                    # 创建新的bbox字符串
                    new_bbox_str = f"[{int(resized_bbox[0])}, {int(resized_bbox[1])}, {int(resized_bbox[2])}, {int(resized_bbox[3])}]"
                    # 替换原始字符串中的bbox
                    old_bbox_str = f"[{x1}, {y1}, {x2}, {y2}]"
                    prompt_text = prompt_text.replace(old_bbox_str, new_bbox_str)

                
                
            # 更新prompts_text
            prompts_text[i] = prompt_text
        
        
        
        # Pass the flat list of images and counts to the processor
        # Check if the processor supports 'image_counts' argument
        processor_signature = inspect.signature(self.processing_class.__call__)
        processor_kwargs = {
            "text": prompts_text,
            "images": flat_images, # Pass the flat list
            "return_tensors": "pt",
            "padding": True,
            "padding_side": "left",
            "add_special_tokens": False,
        }
        if 'image_counts' in processor_signature.parameters:
             processor_kwargs['image_counts'] = image_counts # Pass the counts if supported

        prompt_inputs = self.processing_class(**processor_kwargs)
        prompt_inputs = super()._prepare_inputs(prompt_inputs)


        # Extract input components
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs["pixel_values"]
        image_grid_thw = prompt_inputs["image_grid_thw"]

        # Limit prompt length if needed
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            # Handle special case for Qwen2.5VL models
            if "Qwen2.5-VL" in unwrapped_model.config._name_or_path:
                # Ensure pixel_values is properly formatted
                if len(prompt_inputs['pixel_values'].shape) == 3:
                    prompt_inputs['pixel_values'] = prompt_inputs['pixel_values'].unsqueeze(0)
            # Generate completions
            # The generate function expects batch dimension first.
            # Check how processor handles list of lists input regarding batching.
            # Assuming processor flattens or handles it appropriately for generate.
            # If generate fails, prompt_inputs might need reshaping or different handling.
            prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

            # Split prompt and completion
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

            # Repeat prompt mask for each generation
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Create completion mask by finding the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Combine masks and repeat image tensors
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        pixel_values = pixel_values.repeat_interleave(self.num_generations, dim=0)
        image_grid_thw = image_grid_thw.repeat_interleave(self.num_generations, dim=0)
        # prompt_completion_ids = prompt_completion_ids.repeat_interleave(self.num_generations, dim=0) # Repeat generated ids too


        # Get log probabilities from current model
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        per_token_logps = per_token_logps[:, prompt_length - 1:] # Adjust index if prompt_length definition changes

        # Get log probabilities from reference model
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw
                    )
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:] # Adjust index

        # Calculate KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
             # This assumes completions correspond 1:1 with original batch size * num_generations
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Prepare for reward calculation
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        # --- Start: Prepare reward_kwargs including sizes ---
        reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion", "image"]}
        for key in reward_kwargs:
            for example in inputs:
                reward_kwargs[key].extend([example[key]] * self.num_generations)

        # original_sizes and resized_sizes now correctly have one pair per sample
        reward_kwargs['original_size'] = [size for size in original_sizes for _ in range(self.num_generations)]
        reward_kwargs['resized_size'] = [size for size in resized_sizes for _ in range(self.num_generations)]
        # --- End: Prepare reward_kwargs including sizes ---

        # ... rest of the reward calculation and loss computation ...
        # Calculate rewards for each reward function
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                # Handle model-based reward functions
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]

                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)

                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Handle custom reward functions
                if "solution" in reward_kwargs:
                    solutions = reward_kwargs["solution"]
                else:
                    solutions = [None] * len(prompts)

                reward_kwargs = {key: reward_kwargs[key] for key in reward_kwargs if key != "solution"}
                output_reward_func = reward_func(
                    # prompts=prompts,
                    completions=completions,
                    solution=solutions,
                    **reward_kwargs
                )

                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum rewards from all functions
        rewards = rewards_per_func.sum(dim=1)

        # Calculate grouped rewards statistics
        # Ensure rewards shape matches (batch_size * num_generations)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize rewards to compute advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Calculate GRPO loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)).mean()

        # Log metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())


        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Log training metrics"""
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()



