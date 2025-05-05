export DATA_PATH=/data1/lihaobo/tracking/data/cropped_rft/tracking_dataset
export CKPT_PATH=./share_models/Qwen2.5-VL-3B-Instruct
export SAVE_PATH=./share_models/Qwen2.5-VL-3B-Instruct_GRPO_tracking

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./logs/3b_GRPO_traking.txt"
export CUDA_VISIBLE_DEVICES=0,1,2,3


# python \
#     src/virft/src/open_r1/grpo_tracking.py \
#     --output_dir ${SAVE_PATH}  \
#     --model_name_or_path ${CKPT_PATH} \
#     --dataset_name ${DATA_PATH} \
#     --deepspeed ./src/virft/local_scripts/zero3.json \
#     --max_prompt_length 1024 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --logging_steps 1 \
#     --bf16 \
#     --report_to wandb \
#     --gradient_checkpointing true \
#     --attn_implementation flash_attention_2 \
#     --max_pixels 401408 \
#     --num_train_epochs 2 \
#     --run_name Qwen2_5-VL-3B_GRPO_tracking \
#     --save_steps 100 \
#     --save_only_model true \
#     --num_generations 4


torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/virft/src/open_r1/grpo_tracking.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed ./src/virft/local_scripts/zero3.json \
    --max_prompt_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 102400 \
    --num_train_epochs 2 \
    --run_name Qwen2_5-VL-3B_GRPO_tracking \
    --save_steps 1000 \
    --save_only_model true \
    --num_generations 2

