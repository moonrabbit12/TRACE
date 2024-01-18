#!bin/bash
port=$(shuf -i25000-30000 -n1)
model_name=$"facebook/opt-2.7b"
model=$"opt-2.7b"
dataset=$"Benchmark_5000"
data_path=$"/home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_5000"
clmethod=$"base"
deepspeed --include=localhost:2 --master_port $port inference/infer_single.py  \
    --data_path $data_path \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path $model_name \
    --inference_model_path /mnt/data1/joon/outputs/cl/$clmethod/$model/$dataset \
    --inference_batch 1 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --CL_method base \
    --inference_output_path /mnt/data1/joon/outputs/cl/$clmethod/$model/$dataset > /mnt/data1/joon/outputs/cl/$clmethod/$model/$dataset/infer.log 2>&1 &