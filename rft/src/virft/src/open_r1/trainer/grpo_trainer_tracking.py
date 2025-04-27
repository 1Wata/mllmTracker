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

def simgle_collator(features):
    return features

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen25VLGRPOTrainer(Trainer):
    """
    Trainer for Qwen2.5VL model using Group Relative Policy Optimization (GRPO) method for template-guided object tracking.
    
    此版本专门用于模板引导的单帧目标跟踪，通过多个模板帧引导模型在最后一帧中定位目标。
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
        data_collator = simgle_collator

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

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw=None):
        """
        Calculate per-token log probabilities for the given model and inputs
        """
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }
        if image_grid_thw is not None:
            model_inputs["image_grid_thw"] = image_grid_thw

        outputs = model(**model_inputs)
        logits = outputs.logits  # (B, L, V)
        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]

        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)

        return torch.stack(per_token_logps)

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        """Custom input preparation for GRPO trainer"""
        # prepared_inputs = {}
        # for k, v in inputs.items():
        #     if isinstance(v, torch.Tensor):
        #         prepared_inputs[k] = v.to(self.accelerator.device)
        #     else:
        #         prepared_inputs[k] = v
        # return prepared_inputs

        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute GRPO loss for template-guided object tracking task.
        """
        if return_outputs:
            raise ValueError("The GRPO Trainer does not support returning outputs")

        batch_prompts_structured = [x["prompt"] for x in inputs]
        batch_images = [x["image"] for x in inputs]  # 保留所有图像，包括模板帧和搜索帧
        batch_solutions = [x["solution"] for x in inputs]

        formatted_texts = []
        flattened_images = []
        image_token = getattr(self.processing_class.tokenizer, 'image_token', '<image>')

        for prompt_struct, images_for_sample in zip(batch_prompts_structured, batch_images):
            current_text = ""
            if prompt_struct and isinstance(prompt_struct, list) and 'content' in prompt_struct[0]:
                content_list = prompt_struct[0]['content']
                image_index = 0
                for item in content_list:
                    if item.get('type') == 'image':
                        # 直接使用处理器的方法处理图像，而不是手动添加标记
                        if image_index < len(images_for_sample):
                            flattened_images.append(images_for_sample[image_index])
                            image_index += 1
                            # 使用正确的标记序列
                            current_text += f"<vision_start><image>"
                        else:
                            print(f"Warning: Not enough images provided for this prompt.")
                    elif item.get('text') is not None:
                        current_text += item['text']
            else:
                print(f"Warning: Unexpected prompt structure: {prompt_struct}")

            formatted_texts.append(current_text)

        try:
            prompt_inputs = self.processing_class(
                text=formatted_texts,
                images=flattened_images,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        except Exception as e:
            print("Error during processing:", e)
            print("Formatted Texts:", formatted_texts)
            print("Number of flattened images:", len(flattened_images))
            raise e

        prompt_inputs = self._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs.get("pixel_values")
        image_grid_thw = prompt_inputs.get("image_grid_thw")

        if pixel_values is None:
            raise ValueError("Processor did not return 'pixel_values'. Check processor usage and input.")

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        generation_inputs = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "pixel_values": pixel_values,
        }
        if image_grid_thw is not None:
            generation_inputs["image_grid_thw"] = image_grid_thw

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**generation_inputs, generation_config=self.generation_config)

        prompt_length = prompt_ids.size(1)
        completion_ids = prompt_completion_ids[:, prompt_length:]

        prompt_mask_repeated = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask_full = torch.cat([prompt_mask_repeated, completion_mask], dim=1)

        pixel_values_repeated = pixel_values.repeat_interleave(self.num_generations, dim=0)
        image_grid_thw_repeated = image_grid_thw.repeat_interleave(self.num_generations, dim=0) if image_grid_thw is not None else None

        per_token_logps = self._get_per_token_logps(
            model,
            prompt_completion_ids,
            attention_mask_full,
            pixel_values_repeated,
            image_grid_thw_repeated
        )
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask_full, pixel_values_repeated, image_grid_thw_repeated
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, prompt_completion_ids, attention_mask_full, pixel_values_repeated, image_grid_thw_repeated
                    )
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        completions_decoded = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        completions_structured = [[{"role": "assistant", "content": completion}] for completion in completions_decoded]

        prompts_repeated = [p for p in batch_prompts_structured for _ in range(self.num_generations)]
        solutions_repeated = [s for s in batch_solutions for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts_repeated), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                pass
            else:
                # 调用模板引导的奖励函数
                output_reward_func = reward_func(
                    completions=completions_structured,
                    solution=solutions_repeated,
                    template_guided=True,  # 标明这是模板引导的任务
                )
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)

        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / (completion_mask.sum(dim=1) + 1e-8)).mean()

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
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()