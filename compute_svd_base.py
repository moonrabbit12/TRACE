#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import sys
sys.dont_write_bytecode = True

import pickle

import argparse
import os
import math
import sys
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    get_constant_schedule_with_warmup
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.utils import safe_get_full_grad


sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_prompt_dataset
from utils.data.data_collator import DataCollator
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model, get_latent_directions, generate_basis_pipeline, generate_basis_for_opt, generate_basis_for_bloom

# add flash attention
from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
#from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn

replace_llama_attn_with_flash_attn()
#replace_bloom_attn_with_flash_attn()

# my_peft中修改了lora相关的逻辑
from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import PP, convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model


from model.CustomLlamaForCausalLM import CustomLlamaForCausalLM
from model.CustomOPTForCausalLM import CustomOPTForCausalLM
from model.CustomBloomForCausalLM import CustomBloomForCausalLM

# TODO, check support for OPT and llama

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def none_or_int(value):
    if value == 'None':
        return None
    return int(value)

def parse_args():
    def list_of_strings(arg):
        return arg.split(',')
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        type=str,
                        default='Dahoas/rm-static',
                        help='Path to the training dataset, a single data path.')
    parser.add_argument('--dataset_name',
                        type=list_of_strings,
                        default='all',
                        help='Dataset to be used.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=list_of_strings,
                        default=None,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default=None,
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="A seed for reproducible training.")
    # local_rank 一般表示当前进程在当前节点的编号，global_rank 表示当前进程在所有进程中的编号
    # local_rank 为 -1 时，表示不使用分布式训练。这个值一般由 pytorch/deepspeed 自动设置，用户不用管
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    # store_true 表示如果命令行中有这个参数，则 args.disable_dropout 为 True, 否则默认为 False
    parser.add_argument('--disable_dropout',
                        action='store_true',
                        help='Disable the dropout of the model.')
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument(
        '--zero_stage',
        type=int,
        default=2,
        help='ZeRO optimization stage for Actor model (and clones).')
    
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        action='store_true',
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="step1_tensorboard")
    ## Print loss
    parser.add_argument('--print_loss',
                        action='store_true',
                        help='Prints loss at each step.')
    parser.add_argument('--proj_config_path',
                        type=str,
                        default=None,
                        help='path to proj config pickles')
    # added by wangxiao
    parser.add_argument('--CL_method',
                default=None,
                help='continual learning method used')
    parser.add_argument('--model',
                        default=None,
                        help='name of model')
    parser.add_argument('--repurpose_dim_size',
                        type=int,
                        default=100,
                        help='repurpose dimension size')
    parser.add_argument('--use_repurposed_dims',
                        type=str2bool,
                        help='project to base or dormant subspace')
    parser.add_argument('--ffn_only',
                        type=str2bool,
                        help='Project only ffn layer. Set this flag if you want to enable it.')
    parser.add_argument('--mha_only',
                        type=str2bool,
                        help='Project only mha. Set this flag if you want to enable it.')
    parser.add_argument('--qk_only',
                        type=str2bool,
                        help='Project only qk circuit. Set this flag if you want to enable it.')
    parser.add_argument('--ov_only',
                        type=str2bool,
                        help='Project only ov circuit. Set this flag if you want to enable it.')
    parser.add_argument('--step_size',
                        type=none_or_int,
                        default=None,
                        help='Step size of affine shift')
    print('here is parser')
    print(parser)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    print('here is args in the parse args')
    print(args)

    return args


def main():
    args = parse_args()
    print('hre is the stuffffffffffffff')
    print(args)
    if args.local_rank == -1:
        print(args.local_rank)
        device = torch.device("cuda")
    else:
        print(args.local_rank)
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="v2_sft")
    # set batch size
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
        ) * args.gradient_accumulation_steps

    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    # Barrier to make sure all process are ready to train
    torch.distributed.barrier()

    
    proj_config_path = "/mnt/data1/joon/projconfigs/"
    model_names = ['facebook/opt-1.3b', 'facebook/opt-2.7b', 'bigscience/bloom-560m', 'bigscience/bloom-1b1', 'bigscience/bloom-3b']
    repurposed_dims_size = [10, 100, 500]
    for model_name in model_names:
        for repurposed_dims_size in repurposed_dims_sizes:
            tokenizer = load_hf_tokenizer(model_name, fast_tokenizer=True)
            # default the LLM is decoder only model, so padding side is left
            assert tokenizer.padding_side == 'left'
            assert tokenizer.truncation_side == "left"
            if 'vicuna' in model_name:
                model = create_hf_model(CustomLlamaForCausalLM,
                                        model_name,
                                        tokenizer,
                                        ds_config=ds_config,
                                        disable_dropout=args.disable_dropout,
                                        args=args
                                        )

            elif 'opt' in model_name:
                # TODO: fix this logic so that repetitive code is tightened up for SVD
                model = create_hf_model(CustomOPTForCausalLM,
                                        model_name,
                                        tokenizer,
                                        ds_config=ds_config,
                                        disable_dropout=args.disable_dropout,
                                        args=args
                                        )
            elif 'bloom' in model_name:
                model = create_hf_model(CustomBloomForCausalLM,
                                        model_name,
                                        tokenizer,
                                        ds_config=ds_config,
                                        disable_dropout=args.disable_dropout,
                                        args=args
                                        )
            projection_configs = None
            PROJ_CONFIG_PATH = proj_config_path + model_name.split('/')[1] + '_' + str(repurposed_dims_size) + '_proj_config.pkl'
            if not os.path.exists(PROJ_CONFIG_PATH):
                if 'vicuna' in args.model_name_or_path:
                    projection_config = generate_basis_pipeline(model, repurposed_dims_size)
                elif 'opt' in args.model_name_or_path:
                    projection_configs = generate_basis_for_opt(model, repurposed_dims_size)
                elif 'bloom' in args.model_name_or_path:
                    projection_configs = generate_basis_for_bloom(model, repurposed_dims_size)
                with open(PROJ_CONFIG_PATH, 'wb') as f:
                    pickle.dump(projection_configs, f)
                print("projection configs has been pickled and saved to disk.")

    


if __name__ == "__main__":
    main()
