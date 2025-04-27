import os
import sys
import argparse

# 设置环境变量
os.environ["DATA_PATH"] = "/data1/lihaobo/tracking/rft/tracking/tracking_mini4.json"
os.environ["CKPT_PATH"] = "/data1/lihaobo/tracking/rft/share_models/Qwen2.5-VL-3B-Instruct"
os.environ["SAVE_PATH"] = "/data1/lihaobo/tracking/rft/share_models/Qwen2.5-VL-3B-Instruct_GRPO_tracking"
os.environ["DEBUG_MODE"] = "true"
os.environ["LOG_PATH"] = "/data1/lihaobo/tracking/rft/debug_log/3b_GRPO_traking.txt"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 将命令行参数添加到sys.argv
sys.argv = [
    "src/virft/src/open_r1/grpo_tracking.py",
    "--output_dir", os.environ["SAVE_PATH"],
    "--model_name_or_path", os.environ["CKPT_PATH"],
    "--dataset_name", os.environ["DATA_PATH"],
    "--deepspeed", "/data1/lihaobo/tracking/rft/src/virft/local_scripts/zero3.json",
    "--max_prompt_length", "1024",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--logging_steps", "1",
    "--bf16",
    "--report_to", "wandb",
    "--gradient_checkpointing", "true",
    "--attn_implementation", "flash_attention_2",
    "--max_pixels", "401408",
    "--num_train_epochs", "2",
    "--run_name", "Qwen2_5-VL-3B_GRPO_tracking",
    "--save_steps", "100",
    "--save_only_model", "true",
    "--num_generations", "4"
]

# 导入并运行主脚本
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from virft.src.open_r1.grpo_tracking import main2

if __name__ == "__main__":
    # 这里可以设置断点进行调试
    main2()