#!bin/bash

cl_method="SVD"
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0,1,2,3 --master_port $port training/main.py \
    --data_path /home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path lmsys/vicuna-7b-v1.3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 8 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs 5,3,7,5,3,5,5,7 \
    --gradient_accumulation_steps 32 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed 1234 \
    --zero_stage 3 \
    --deepspeed \
    --gradient_checkpointing $true \
    --print_loss \
    --CL_method $cl_method \
    --output_dir /home/joon/TRACE/outputs/cl/$cl_method/vicuna7b > /home/joon/TRACE/outputs/cl/$cl_method/vicuna7b/train.log 2>&1 &