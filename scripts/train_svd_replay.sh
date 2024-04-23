#!bin/bash
clmethod="svd_replay"
CL_method="svd_replay"
port=$(shuf -i25000-30000 -n1)
#model_name=$"microsoft/phi-1_5"
#model=$"phi-1_5"
model_name=$"facebook/opt-1.3b"
model=$"opt-1.3b"
dataset=$"Benchmark_500"
data_path=$"/home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500"
proj_config_path=$"/mnt/data1/joon/projconfigs/"
repurpose_dim_size=128
seed=1234
step_size=15
use_repurposed_dims=false
project_only_first_layer=false
ffn_only=false
mha_only=false
qk_only=false
ov_only=true
ablation_name="full"
layer_ablation="all"

if [ $ffn_only == true ] ; then
    ablation_name="ffn_only"
fi
if [ $mha_only == true ] ; then
    ablation_name="mha_only"
fi
if [ $qk_only == true ] ; then
    ablation_name="qk_only"
fi
if [ $ov_only == true ] ; then
    ablation_name="ov_only"
fi
if [ $use_repurposed_dims == true ] ; then
    repurpose="dormant"
fi
if [[ $step_size != "None" && $use_repurposed_dims == false ]] ; then
    repurpose="affine/${step_size}"
fi

output_dir=$"/mnt/data1/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset/$seed/$repurpose/$ablation_name"
log_file=$"/mnt/data1/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset/$seed/$repurpose/$ablation_name/train.log"

if [ $project_only_first_layer == true ] ; then
    layer_ablation="first_only"
    output_dir=$"/mnt/data1/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset/$seed/$repurpose/$ablation_name/$layer_ablation"
    log_file=$"/mnt/data1/joon/outputs/cl/$clmethod/$model/$repurpose_dim_size/$dataset/$seed/$repurpose/$ablation_name/$layer_ablation/train.log"
fi
mkdir -p $output_dir
echo $repurpose
echo $use_repurposed_dims
deepspeed --include=localhost:4,5,6,7 --master_port $port training/svd_replay.py  \
    --data_path /home/joon/TRACE/TRACE-Benchmark/LLM-CL-Benchmark_500 \
    --dataset_name C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --replay_dataset_name Lima \
    --model_name_or_path $model_name \
    --model $model \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --max_prompt_len 1024 \
    --max_ans_len 512 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
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
    --CL_method $CL_method \
    --repurpose_dim_size $repurpose_dim_size \
    --project_only_first_layer $project_only_first_layer \
    --ffn_only $ffn_only \
    --mha_only $mha_only \
    --qk_only $qk_only \
    --ov_only $ov_only \
    --proj_config_path $proj_config_path \
    --step_size $step_size \
    --past_task_ratio 0.1 \
    --output_dir $output_dir > $log_file 2>&1 &
