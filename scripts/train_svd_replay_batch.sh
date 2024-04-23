#!/bin/bash
# Constant Parameters
cl_method="svd_replay"
clmethod="svd_replay"
#model_name=$"bigscience/bloom-1b1"
#model=$"bloom-1b1"
model_name=$"facebook/opt-1.3b"
model=$"opt-1.3b"
#model_name=$"microsoft/phi-1_5"
#model=$"phi-1_5"
dataset=$"Benchmark_500"
data_path=$"/home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500"
proj_config_path=$"/mnt/data1/joon/projconfigs/"
port=$(shuf -i25000-30000 -n1)

# Grid Search Parameters
ablation_names=("ov_only" "mha_only" "qk_only" "ffn_only" "full")
repurpose_dim_sizes=(128)  
step_sizes=(0 1 5 10 15 20 25 50)              
seeds=(1331 42 1234)              
layer_projs=(false)
# Iterate over grid search parameters
for ablation_name in ${ablation_names[@]}; do
    for repurpose_dim_size in ${repurpose_dim_sizes[@]}; do
        for step_size in ${step_sizes[@]}; do
            for seed in ${seeds[@]}; do
                for proj in ${layer_projs[@]}; do
                    # Set boolean flags based on ablation_name
                    ffn_only=false
                    mha_only=false
                    qk_only=false
                    ov_only=false
                    use_repurposed_dims=false
                    project_only_first_layer=$proj
                    case $ablation_name in
                        ffn_only)
                            ffn_only=true
                            ;;
                        mha_only)
                            mha_only=true
                            ;;
                        qk_only)
                            qk_only=true
                            ;;
                        ov_only)
                            ov_only=true
                            ;;
                    esac

                    # Determine repurpose type
                    repurpose="affine/${step_size}"
                    
            

                    # Output and Log Paths
                    output_dir="/mnt/data4/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset/$seed/$repurpose/$ablation_name"
                    log_file="$output_dir/train.log"
                    if [ $project_only_first_layer == true ] ; then
                        layer_ablation="first_only"
                        output_dir=$"/mnt/data4/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset/$seed/$repurpose/$ablation_name/$layer_ablation"
                        log_file=$"/mnt/data4/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset/$seed/$repurpose/$ablation_name/$layer_ablation/train.log"
                    fi
                    mkdir -p $output_dir

                    
                    # Check if "train.log" exists in the directory
                    if [ -f "$log_file" ]; then
                        echo "train.log already exists in $output_dir. Skipping this iteration."
                        continue  # Skip this iteration
                    fi
                    echo "Starting training with config: Ablation: $ablation_name, Dim Size: $repurpose_dim_size, Step: $step_size, Seed: $seed"

                    deepspeed --include=localhost:0,3 training/svd_replay.py \
                        --data_path /home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500 \
                        --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
                        --replay_dataset_name Lima \
                        --model_name_or_path $model_name \
                        --model $model \
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
                        --use_repurposed_dims $use_repurposed_dims \
                        --CL_method $cl_method \
                        --repurpose_dim_size $repurpose_dim_size \
                        --project_only_first_layer $project_only_first_layer \
                        --ffn_only $ffn_only \
                        --mha_only $mha_only \
                        --qk_only $qk_only \
                        --ov_only $ov_only \
                        --proj_config_path $proj_config_path \
                        --step_size $step_size \
                        --past_task_ratio 0.1 \
                        --output_dir $output_dir > $log_file 2>&1
                    
                    echo "Training completed for $ablation_name, logs at $log_file"
                done
            done
        done
    done
done

echo "All grid search experiments completed"
