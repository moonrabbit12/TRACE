#!bin/bash
seed=1234
port=$(shuf -i25000-30000 -n1)
#model_name=$"microsoft/phi-1_5"
#model=$"phi-1_5"
model_name=$"facebook/opt-1.3b"
model=$"opt-1.3b"
dataset=$"Benchmark_500"
data_path=$"/home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500"
deepspeed --include=localhost:0,1 --master_port $port training/svd_replay.py  \
    --data_path /home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --replay_dataset_name Lima \
    --model_name_or_path $model_name \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --num_train_epochs 5,3,7,5,3,5,5,7 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 0 \
    --seed $seed \
    --zero_stage 2 \
    --deepspeed \
    --gradient_checkpointing $true \
    --print_loss \
    --past_task_ratio 0.1 \
    --output_dir /mnt/data1/joon/outputs/cl/replay/$model/$dataset/$seed > /mnt/data1/joon/outputs/cl/replay/$model/$dataset/$seed/train.log 2>&1 &