#!/bin/bash
port=$(shuf -i25000-30000 -n1)
model_name=$"facebook/opt-1.3b"
model=$"opt-1.3b"
#model_name=$"bigscience/bloom-1b1"
#model=$"bloom-1b1"
#model_name=$"microsoft/phi-1_5"
#model=$"phi-1_5"
dataset=$"Benchmark_500"
data_path=$"/home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500"
proj_config_path=$"/mnt/data1/joon/projconfigs/"

clmethod=$"svd_replay"
layer_ablation="all"

# Grid Search Parameters
ablation_names=("ffn_only")
#ablation_names=("ffn_only")
repurpose_dim_sizes=(128)  
step_sizes=(0 1 5 10 15 20 25 50)              
#step_sizes=(0)              
seeds=(1331 42 1234)              
#seeds=(1331)              
layer_projs=(false)
projinfs=("SVD")
# Iterate over grid search parameters
for ablation_name in ${ablation_names[@]}; do
    for repurpose_dim_size in ${repurpose_dim_sizes[@]}; do
        for step_size in ${step_sizes[@]}; do
            for seed in ${seeds[@]}; do
                for proj in ${layer_projs[@]}; do
                    for projinf in ${projinfs[@]}; do
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
                        output_dir=$"/mnt/data4/joon/outputs/cl/${clmethod}/${model}/${repurpose_dim_size}/${dataset}/${seed}/${repurpose}/${ablation_name}/${projinf}"
                        log_file="$output_dir/infer.log"
                        inf_model_path=$"/mnt/data4/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset/$seed/$repurpose/$ablation_name"
                        if [ $project_only_first_layer == true ] ; then
                            layer_ablation="first_only"
                            output_dir=$"/mnt/data4/joon/outputs/cl/${clmethod}/${model}/${repurpose_dim_size}/${dataset}/${seed}/${repurpose}/${ablation_name}/${layer_ablation}/${projinf}"
                            log_file="$output_dir/infer.log"
                            inf_model_path=$"/mnt/data4/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset/$seed/$repurpose/$ablation_name/$layer_ablation"
                        fi
                        mkdir -p $output_dir

                        
                        # Check if "train.log" exists in the directory
                        if [ -f "$log_file" ]; then
                            echo "infer.log already exists in $output_dir. Skipping this iteration."
                            continue  # Skip this iteration
                        fi
                        echo "Starting inference with config: Ablation: $ablation_name, Dim Size: $repurpose_dim_size, Step: $step_size, Seed: $seed"

                        deepspeed --include=localhost:3 --master_port $port inference/infer_single.py  \
                            --data_path $data_path \
                            --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
                            --model_name_or_path $model_name \
                            --inference_model_path $inf_model_path \
                            --inference_batch 8 \
                            --max_prompt_len 1024 \
                            --max_ans_len 512 \
                            --seed $seed \
                            --deepspeed \
                            --CL_method $projinf \
                            --use_repurposed_dims $use_repurposed_dims \
                            --model $model \
                            --repurpose_dim_size $repurpose_dim_size \
                            --ffn_only $ffn_only \
                            --mha_only $mha_only \
                            --qk_only $qk_only \
                            --ov_only $ov_only \
                            --proj_config_path $proj_config_path \
                            --step_size $step_size \
                            --inference_output_path $output_dir > $log_file 2>&1
                        
                        echo "Inference completed for $ablation_name, logs at $log_file"
                    done
                done
            done
        done
    done
done

echo "All grid search experiments completed"
