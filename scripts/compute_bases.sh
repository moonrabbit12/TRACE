#!/bin/bash
# Constant Parameters
cl_method="SVD"

dataset=$"Benchmark_500"
data_path=$"/home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500"
proj_config_path=$"/mnt/data1/joon/projconfigs/"
port=$(shuf -i25000-30000 -n1)

# Grid Search Parameters
model_names=("facebook/opt-2.7b")
repurpose_dim_sizes=(10 100 500)  # Example values
seed=42
step_size=30
ffn_only=false
mha_only=false
qk_only=false
ov_only=false
use_repurposed_dims=false
ablation_name="full"
repurpose="affine/${step_size}"
# Iterate over grid search parameters
for model_name in ${model_names[@]}; do
    model="${model_name#*/}"
    echo "$model"
    for repurpose_dim_size in ${repurpose_dim_sizes[@]}; do
        # Set boolean flags based on ablation_name

        # Output and Log Paths
        output_dir="/mnt/data1/joon/outputs/cl/$cl_method/$model/$repurpose_dim_size/$dataset/$seed/$repurpose/$ablation_name"
        log_file="$output_dir/train.log"

        echo "Starting training with config: Ablation: $ablation_name, Dim Size: $repurpose_dim_size, Step: $step_size, Seed: $seed"

        deepspeed --include=localhost:2 training/main.py \
            --data_path $data_path \
            --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
            --model_name_or_path $model_name \
            --model $model \
            --per_device_train_batch_size 16 \
            --per_device_eval_batch_size 4 \
            --max_prompt_len 1024 \
            --max_ans_len 512 \
            --learning_rate 1e-5 \
            --weight_decay 0.0 \
            --num_train_epochs 5,3,7,5,3,5,5,7 \
            --gradient_accumulation_steps 2 \
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
            --ffn_only $ffn_only \
            --mha_only $mha_only \
            --qk_only $qk_only \
            --ov_only $ov_only \
            --proj_config_path $proj_config_path \
            --step_size $step_size \
            --output_dir $output_dir > $log_file 2>&1
        
        echo "Training completed for $ablation_name, logs at $log_file"


    done
done

echo "All grid search experiments completed"
