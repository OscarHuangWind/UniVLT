export WANDB_PROJECT="UniVLT"
export WANDB_NAME="stage1_fine_tune"
WANDB_API_KEY=your_wandb_api_key
wandb login --relogin $WANDB_API_KEY
CUDA_VISIBLE_DEVICES=0,1,2,3 \
MASTER_PORT=29502 \
NPROC_PER_NODE=3 \
swift sft \
    --model model_ckpts/stage1_fine_tune \
    --train_type lora \
    --torch_dtype bfloat16 \
    --target_modules all-linear \
    --dataset '/mnt/data2/LTA/maxPixel112896x2_single_image_conversations_TRAIN_combined.json' \
              '/mnt/data2/LTA/TRAIN_3_image_risk_id_3000risk_1000noRisk_UNIQUE.json' \
              '/mnt/data2/LTA/TRAIN_3_image_risk_id_analysis_3000risk_1000noRisk_UNIQUE-open-ended.json' \
              "/mnt/data2/omnidrive/json/action_train.json#1500" \
              "/mnt/data2/omnidrive/json/perception_train.json#1500" \
              "/mnt/data2/CODA/general_perception_data.json#1000" \
              "/mnt/data2/CODA/explanation_data_coda_single_remapped.json#1000" \
              "/mnt/data2/CODA/general_suggestion_data.json#1000" \
              "/mnt/data2/LingoQA/json/training_data.json#3000" \
    --lora_rank 8 \
    --lora_alpha 32 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 2 \
    --eval_steps -1 \
    --save_steps 200 \
    --save_total_limit 10 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir model_ckpts/stage2_fine_tune \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --deepspeed zero2 \
    --report_to wandb \
    --model_kwargs '{"max_pixels": 225792}' \
    --attn_impl flash_attn