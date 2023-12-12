#!bin/bash
port=$(shuf -i25000-30000 -n1)
deepspeed --include=localhost:0 --master_port $port inference/infer_single.py \
   --data_path /home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_5000 \
   --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
   --model_name_or_path lmsys/vicuna-7b-v1.3 \
   --inference_model_path /home/joon/TRACE/outputs/lora \
   --inference_batch 1 \
   --max_prompt_len 1024 \
   --max_ans_len 512 \
   --seed 1234 \
   --deepspeed \
   --CL_method lora \
   --inference_output_path /home/joon/TRACE/outputs/lora/predictions > /home/joon/TRACE/outputs/lora/infer.log 2>&1 &
   
