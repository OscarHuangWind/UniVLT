export WANDB_PROJECT="UniVLT"
export WANDB_NAME="stage1_fine_tune"
WANDB_API_KEY=your_wandb_api_key
wandb login --relogin $WANDB_API_KEY
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_PORT=29501 \
NPROC_PER_NODE=4 \
swift sft \
    --model /path/to/base/model \
    --train_type lora \
    --torch_dtype bfloat16 \
    --target_modules all-linear \
    --dataset "/path/to/LingoQA/json/training_data.json" \
    "/path/to/CODA/general_perception_data.json" \
    "/path/to/CODA/general_suggestion_data.json" \
    "/path/to/CODA/region_perception_data.json" \
    "/path/to/omnidrive/json/action_train.json" \
    "/path/to/omnidrive/json/perception_train.json" \
    --lora_rank 8 \
    --lora_alpha 32 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --eval_steps -1 \
    --save_steps 1000 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir model_ckpts/stage1_fine_tune \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero2 \
    --report_to wandb \
    --model_kwargs '{"max_pixels": 225792}' \
    --attn_impl flash_attn