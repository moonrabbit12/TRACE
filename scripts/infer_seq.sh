#!bin/bash
port=$(shuf -i25000-30000 -n1)
model_name=$"facebook/opt-2.7b"
model=$"opt-2.7b"
dataset=$"Benchmark_500"
data_path=$"/home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500"
clmethod=$"SVD"
repurpose_dim_size=500
deepspeed --include=localhost:3 --master_port $port inference/infer_single.py  \
    --data_path $data_path \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --model_name_or_path $model_name \
    --inference_model_path /mnt/data1/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset \
    --inference_batch 16 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --seed 1234 \
    --deepspeed \
    --CL_method base \
    --inference_output_path /mnt/data1/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset > /mnt/data1/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset/infer.log 2>&1 &