#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --account=rrg-mageed
#SBATCH --job-name=alpaca
#SBATCH --output=./outfile/alpaca.out
#SBATCH --error=./outfile/alpaca.err
#SBATCH --mail-user=zcy94@outlook.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

NPROC_PER_NODE=4

module load python/3.10
module load gcc/9.3.0 arrow
module load cuda/11.1.1 cudnn ffmpeg/4.3.2

source ~/alpaca/bin/activate

deepspeed --num_gpus=$NPROC_PER_NODE --num_nodes=1 train.py \
    --model_name_or_path ./eyad-27/ \
    --data_path ./clean_ar_alpaca.json \
    --output_dir ./ckpt_ar_alpaca/ \
    --cache_dir ./cache/ \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --gradient_checkpointing \
    --report_to="none" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 100 \
    --deepspeed "dc_config2.json"
