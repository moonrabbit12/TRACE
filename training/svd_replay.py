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
import random

import numpy as np 

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, ConcatDataset, Sampler
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
from utils.model.model_utils import create_hf_model

# add flash attention
from utils.flash_attention.llama_flash_att import replace_llama_attn_with_flash_attn
#from utils.flash_attention.bloom_flash_att import replace_bloom_attn_with_flash_attn

replace_llama_attn_with_flash_attn()
#replace_bloom_attn_with_flash_attn()

# my_peft中修改了lora相关的逻辑
from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import PP, convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model


from params import Method2Class, AllDatasetName

from model.CustomLlamaForCausalLM import CustomLlamaForCausalLM
from model.CustomOPTForCausalLM import CustomOPTForCausalLM
from model.CustomBloomForCausalLM import CustomBloomForCausalLM

class ShuffledHomogeneousBatchSampler(Sampler):
    def __init__(self, concat_dataset, batch_size):
        self.concat_dataset = concat_dataset
        self.batch_size = batch_size
        
        # Calculate the start index of each dataset in the concatenated dataset
        self.dataset_offsets = [0] + torch.cumsum(torch.tensor([len(d) for d in concat_dataset.datasets]), 0).tolist()[:-1]
        self.i_tasks = []

    def __iter__(self):
        all_batches = []
        for dataset_idx, dataset in enumerate(self.concat_dataset.datasets):
            dataset_start = self.dataset_offsets[dataset_idx]
            indices = torch.randperm(len(dataset)) + dataset_start
            # Create batches for the current dataset
            for i in range(0, len(indices), self.batch_size):
                all_batches.append(indices[i:i+self.batch_size].tolist())
                self.i_tasks.append(dataset_idx)

        # Shuffle the list of all batches to randomize the order of datasets in each epoch
        all_batches, self.i_tasks = unison_shuffled_copies(all_batches, self.i_tasks)
        #np.random.shuffle(all_batches)

        # Yield batches one by one
        for batch in all_batches:
            yield from batch

    def __len__(self):
        # This will give the total number of batches across all datasets
        total_samples = sum(len(d) for d in self.concat_dataset.datasets)
        return (total_samples + self.batch_size - 1) // self.batch_size

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    c = list(zip(a, b))
    random.shuffle(c)
    a, b = zip(*c)
    return a, b

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

# TODO, check support for OPT and llama


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
    
    parser.add_argument('--replay_dataset_name',
                    type=str,
                    default='Lima',
                    help='Dataset to be used.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/tmp/data_files/',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
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
        default=0,
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
    # added by wangxiao
    parser.add_argument('--past_task_ratio',
                default=None,
                help='Replay ratio used for past task')

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
    parser.add_argument('--project_only_first_layer',
                        type=str2bool,
                        help='Project only first transformer block. Set this flag if you want to enable it.')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()


    return args


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
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

    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"


    if 'vicuna' in args.model_name_or_path:
            model = create_hf_model(CustomLlamaForCausalLM,
                                    args.model_name_or_path,
                                    tokenizer,
                                    ds_config=ds_config,
                                    disable_dropout=args.disable_dropout,
                                    args=args
                                    )
    elif 'opt' in args.model_name_or_path:
        model = create_hf_model(CustomOPTForCausalLM,
                                args.model_name_or_path,
                                tokenizer,
                                ds_config=ds_config,
                                disable_dropout=args.disable_dropout,
                                args=args
                                )
    elif 'bloom' in args.model_name_or_path:
        model = create_hf_model(CustomBloomForCausalLM,
                                args.model_name_or_path,
                                tokenizer,
                                ds_config=ds_config,
                                disable_dropout=args.disable_dropout,
                                args=args
                                )
    else:
        model = create_hf_model(AutoModelForCausalLM,
                                args.model_name_or_path,
                                tokenizer,
                                ds_config=ds_config,
                                disable_dropout=args.disable_dropout
                                )

    repurposed_dims_size = args.repurpose_dim_size
    projection_configs = None
    PROJ_CONFIG_PATH = args.proj_config_path + args.model + '_' + str(args.repurpose_dim_size) + '_pca' + '_proj_config.pkl'
    if not os.path.exists(PROJ_CONFIG_PATH):
        if 'vicuna' in args.model_name_or_path:
            projection_config = generate_basis_pipeline(model, repurposed_dims_size)
        elif 'opt' in args.model_name_or_path:
            projection_configs = generate_basis_for_opt(model, repurposed_dims_size)
        elif 'bloom' in args.model_name_or_path:
            projection_configs = generate_basis_for_bloom(model, repurposed_dims_size)
        elif 'phi' in args.model_name_or_path:
            projection_configs = generate_basis_for_phi(model, repurposed_dims_size)
        with open(PROJ_CONFIG_PATH, 'wb') as f:
            pickle.dump(projection_configs, f)
        print("projection configs has been pickled and saved to disk.")
    else:
        with open(PROJ_CONFIG_PATH, 'rb') as f:
            projection_configs = pickle.load(f)
        print("projection configs has been loaded from disk.")

    projection_configs = tuple(
        [(a.to(device), b.to(device), c.to(device)) for a,b,c in config_list] for config_list in projection_configs
    )

    train_task_list = {}
    eval_task_list = {}
    test_task_list = {}
    
    replay_dataset_list={}

    def get_dataset(dataset):
        dataset_path = os.path.join(args.data_path,dataset)
        # Prepare the data
        if dataset==args.replay_dataset_name:
            sample_ratio=None
        else:
            sample_ratio=eval(args.past_task_ratio)
        replay_dataset, _, _ = create_prompt_dataset(
            args.local_rank,
            dataset_path,
            args.data_output_path,
            args.seed,
            sample_ratio=sample_ratio
        )
        train_dataset, eval_dataset, test_dataset = create_prompt_dataset(
            args.local_rank,
            dataset_path,
            args.data_output_path,
            args.seed,
        )
        
        # DataLoaders creation:
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_dataset)
            eval_sampler = SequentialSampler(eval_dataset)
            test_sampler = SequentialSampler(test_dataset)
        else:
            train_sampler = DistributedSampler(train_dataset)
            eval_sampler = DistributedSampler(eval_dataset)
            test_sampler = DistributedSampler(test_dataset)


        data_collator  = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=False
        )
        inf_data_collator = DataCollator(
            tokenizer,
            model=model,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=True
        )
                

        train_dataloader = DataLoader(train_dataset,
                                    collate_fn=data_collator,
                                    sampler=train_sampler,
                                    batch_size=args.per_device_train_batch_size)

        eval_dataloader = DataLoader(eval_dataset,
                                    collate_fn=data_collator,
                                    sampler=eval_sampler,
                                    batch_size=args.per_device_eval_batch_size)
        test_dataloader = DataLoader(test_dataset,
                            collate_fn=inf_data_collator,
                            sampler=test_sampler,
                            batch_size=args.per_device_eval_batch_size)
        return train_dataloader, replay_dataset, eval_dataloader, test_dataloader
    
    replay_dataloader,replay_dataset,_,_ = get_dataset(args.replay_dataset_name)
    replay_dataset_list[args.replay_dataset_name] = replay_dataset

    if args.dataset_name[0] == "all":
        Datasets = AllDatasetName
    else:
        Datasets = args.dataset_name
    for dataset in Datasets:
        train_dataloader, replay_dataset, eval_dataloader, test_dataloader = get_dataset(dataset)
 
        train_task_list[dataset] = train_dataloader
        eval_task_list[dataset] = eval_dataloader
        test_task_list[dataset] = test_dataloader
        replay_dataset_list[dataset] = replay_dataset

    def get_optimizer(model):
        # Split weights in two groups, one with weight decay and the other not.
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model, args.weight_decay)

        AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
        optimizer = AdamOptimizer(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                betas=(0.9, 0.95))
        
        total_train_dataloader_len = sum(len(train_task_list[task]) for task in list(train_task_list.keys()))
        num_update_steps_per_epoch = math.ceil(
            total_train_dataloader_len / args.gradient_accumulation_steps)
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps
        )
        
        return optimizer, lr_scheduler
                    
    optimizer, lr_scheduler = get_optimizer(model)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    # print_rank_0(
    #     f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #     args.global_rank)
    # perplexity = evaluation(model, eval_dataloader)
    # print_rank_0(f"ppl: {perplexity}", args.global_rank)

    # Initialize the global progress bar
    def train_one_task(task, i_task, epochs, projection_configs):
        
        #### TRAIN ####
        train_dataloader = train_task_list[task]
        eval_dataloader = eval_task_list[task]
        total_steps = epochs * len(train_dataloader)
        progress_bar = tqdm(total=total_steps, leave=True, disable=(args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                args.global_rank)
            model.train()

            for step, batch in enumerate(train_dataloader):
                del batch['sources']
                batch = to_device(batch, device)
                outputs = model(**batch, projection_configs=projection_configs, use_cache=False, i_task=i_task)
                loss = outputs.loss
                # Update the description to include current step and loss, if needed
                if args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                model.step()

    def replay(i_task, epochs):
        replay_datasets = [replay_dataset_list[Datasets[i]] for i in range(i_task)]
        replay_datasets.append(replay_dataset_list[args.replay_dataset_name])
        replay_datasets = ConcatDataset(replay_datasets)
        #replay_sampler = RandomSampler(replay_datasets)
        replay_sampler = ShuffledHomogeneousBatchSampler(replay_datasets, args.per_device_train_batch_size)
        
        data_collator  = DataCollator(
            tokenizer,
            padding="longest",
            max_prompt_len=args.max_prompt_len,
            max_ans_len=args.max_ans_len,
            pad_to_multiple_of=8,
            inference=False
        )
        replay_dataloader = DataLoader(replay_datasets,
                                    collate_fn=data_collator,
                                    sampler=replay_sampler,
                                    batch_size=args.per_device_train_batch_size)
        if args.local_rank == -1:
            device = torch.device("cuda")
        else:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
        
        #### TRAIN ####
        print("Replaying....................................")

        total_steps = epochs * len(replay_dataloader) * args.per_device_train_batch_size
        progress_bar = tqdm(total=total_steps, leave=True, disable=(args.global_rank != 0))
        for epoch in range(epochs):
            print_rank_0(
                f"Beginning of Epoch {epoch+1}/{epochs}, Total Micro Batches {len(train_dataloader)}",
                args.global_rank)
            model.train()

            for step, batch in enumerate(replay_dataloader):
                i_task = replay_dataloader.sampler.i_tasks[step]
                del batch['sources']
                batch = to_device(batch, device)
                outputs = model(**batch, projection_configs=projection_configs, use_cache=False, i_task=i_task)
                loss = outputs.loss
                # Update the description to include current step and loss, if needed
                if args.global_rank == 0:
                    # Update the progress bar
                    progress_bar.update(1)
                    description = f"Epoch {epoch+1}, Step {step}, Loss: {loss.item():.4f}"
                    progress_bar.set_description(description, refresh=False)

                model.backward(loss)
                # Correct gradient accumulation steps are handled withing the deepspeed engine's backward call.
                model.step()
                
    def save_model(round):
        if args.output_dir is not None:
            print_rank_0('saving model ...', args.global_rank)

        if args.global_rank == 0:
            save_hf_format(model, tokenizer, args, sub_folder=str(round))

        if args.zero_stage == 3:
            # For zero stage 3, each gpu only has a part of the model, so we need a special save function
            save_zero_three_model(model,
                                  args.global_rank,
                                  args.output_dir,
                                  zero_stage=args.zero_stage)
        print_rank_0('Sucessful saving model after round {}'.format(round), args.global_rank)


    for i_task, task in enumerate(train_task_list):
        train_one_task(task, i_task, int(args.num_train_epochs[i_task]), projection_configs)
        replay(i_task, 1)
        save_model(i_task)
        # CL_Trainer.save_model()
        


if __name__ == "__main__":
    main()
