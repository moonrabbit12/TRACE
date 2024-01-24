#!bin/bash

cl_method="SVD"
#model_name=$"bigscience/bloom-1b1"
#model=$"bloom-1b1"
model_name=$"facebook/opt-1.3b"
model=$"opt-1.3b"
dataset=$"Benchmark_500"
data_path=$"/home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500"
port=$(shuf -i25000-30000 -n1)
repurpose_dim_size=100
seed=1234
use_repurposed_dims="true"
if [ $use_repurposed_dims = "true" ]; then
    repurpose="dormant"
else
    repurpose="base"
fi
echo $repurpose
echo $use_repurposed_dims
deepspeed --include=localhost:4,5 --master_port $port training/main.py \
    --data_path $data_path \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path $model_name \
    --model $model \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs 5,3,7,5,3,5,5,7 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed $seed \
    --zero_stage 2 \
    --deepspeed \
    --gradient_checkpointing $true \
    --print_loss \
    --use_repurposed_dims $use_repurposed_dims \
    --CL_method $cl_method \
    --repurpose_dim_size $repurpose_dim_size \
    --output_dir /mnt/data1/joon/outputs/cl/$cl_method/$model/$repurpose_dim_size/$dataset/$seed/$repurpose > /mnt/data1/joon/outputs/cl/$cl_method/$model/$repurpose_dim_size/$dataset/$seed/$repurpose/train.log 2>&1 &