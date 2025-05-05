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

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute GRPO loss for tracking task
        
        Args:
            model: The model being trained
            inputs: Input data dictionary
            return_outputs: Whether to return outputs (not supported)
            num_items_in_batch: Number of items in batch
            
        Returns:
            The computed loss value
        """
        if return_outputs:
            raise ValueError("The GRPO Trainer does not support returning outputs")
        

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        images = [x["image"] for x in inputs]


        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        
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
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        pixel_values = pixel_values.repeat_interleave(self.num_generations, dim=0)
        image_grid_thw = image_grid_thw.repeat_interleave(self.num_generations, dim=0)

        # Get log probabilities from current model
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

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
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        # Calculate KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Prepare for reward calculation
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

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
                # Gather additional kwargs from inputs
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                
                # Special handling for tracking rewards
                if "solution" in reward_kwargs:
                    solutions = reward_kwargs["solution"]
                else:
                    solutions = [None] * len(prompts)
                
                # Special handling for tracking_confidence reward which might need history
                if hasattr(reward_func, "__name__") and "tracking_confidence" in reward_func.__name__:
                    if "tracking_history" in reward_kwargs:
                        tracking_history = reward_kwargs["tracking_history"]
                        output_reward_func = reward_func(
                            prompts=prompts, 
                            completions=completions, 
                            solution=solutions, 
                            tracking_history=tracking_history
                        )
                    else:
                        output_reward_func = reward_func(
                            prompts=prompts, 
                            completions=completions, 
                            solution=solutions
                        )
                else:
                    output_reward_func = reward_func(
                        prompts=prompts, 
                        completions=completions, 
                        solution=solutions, 
                        **{k: v for k, v in reward_kwargs.items() if k != "solution"}
                    )
                
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Sum rewards from all functions
        rewards = rewards_per_func.sum(dim=1)

        # Calculate grouped rewards statistics
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





def custom_tracking_data_collator(features):

    for item in features:
        if "image" in item:
            if isinstance(item["image"], list):
                # 处理图像路径列表
                item["image"] = [Image.open(img_path).convert("RGB") for img_path in item["image"]]
            elif isinstance(item["image"], str):
                # 处理单个图像路径
                item["image"] = Image.open(item["image"]).convert("RGB")
    
    return features