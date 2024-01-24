#!bin/bash
port=$(shuf -i25000-30000 -n1)
model_name=$"bigscience/bloom-3b"
model=$"bloom-3b"
dataset=$"Benchmark_500"
data_path=$"/home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500"
clmethod=$"base"
seed=1234
deepspeed --include=localhost:0,1,2,3 --master_port $port training/main.py  \
    --data_path $data_path \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path $model_name \
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
    --seed $seed \
    --zero_stage 2 \
    --deepspeed \
    --print_loss \
    --CL_method $clmethod \
    --output_dir /mnt/data1/joon/outputs/cl/$clmethod/$model/$dataset/$seed > /mnt/data1/joon/outputs/cl/$clmethod/$model/$dataset/$seed/train.log 2>&1 &


