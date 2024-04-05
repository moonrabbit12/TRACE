"""
    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
"""

# !/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import sys
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import deepspeed
import json
import pickle


from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_collator import DataCollator
from utils.data.data_utils import create_prompt_dataset
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from utils.ds_utils import get_train_ds_config
from utils.model.model_utils import create_hf_model
from evaluations import eval_ScienceQA, eval_MeetingBank, eval_PapyrusF, eval_CStance, eval_Py150, eval_FOMC, eval_NumGLUE_cm, eval_NumGLUE_ds, eval_20Minuten # to be continued
from training.params import Method2Class, AllDatasetName

from model.Replay.LFPT5 import getInitialPrompt
from model.Dynamic_network.PP import PP, convert_PP_model
from model.Dynamic_network.L2P import convert_L2P_model

from model.CustomLlamaForCausalLM import CustomLlamaForCausalLM
from model.CustomOPTForCausalLM import CustomOPTForCausalLM
from model.CustomBloomForCausalLM import CustomBloomForCausalLM

# dist.init_process_group(backend='nccl')

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
                        help='Path to the training dataset. A single data path.')
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
        "--inference_model_path",
        type=str,
        help=
        "Path to inference model.",
        required=True,
    )
    parser.add_argument(
        "--max_prompt_len",
        type=int,
        default=512,
        help="The maximum sequence length.",
    )
    # inference params
    parser.add_argument(
        "--max_ans_len",
        type=int,
        default=256,
        help="The maximum answer length.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generate temperature params.",
    )

    parser.add_argument(
        "--inference_batch",
        type=int,
        default=4,
        help="Inference batch size.",
    )
    # TODO, add other inference params
    parser.add_argument(
        "--inference_tasks",
        type=list_of_strings,
        default='all',
        help='Datasets to be used.'
    )
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

    # added by wangxiao
    parser.add_argument('--inference_output_path',
                        type=str,
                        default=None,
                        help="Where to store inference results.")
    parser.add_argument('--CL_method',
            default=None,
            help='continual learning method used')
    parser.add_argument('--use_repurposed_dims',
                        type=bool, 
                        default=True,
                        help='project to base or dormant subspace')
    parser.add_argument('--repurpose_dim_size',
                        type=int,
                        default=100,
                        help='repurpose dimension size')
    parser.add_argument('--model',
                        default=None,
                        help='name of model')
    parser.add_argument('--proj_config_path',
                        type=str,
                        default=None,
                        help='path to proj config pickles')
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
    set_random_seed(args.seed)
    device = torch.device("cuda")


    def prediction(model, infer_dataloader, inference_task_id, projection_configs):
        predicted_sequences = []
        sources_sequences = []
        ground_truths = []
        model.eval()
        model.model.decoder.i_task = inference_task_id
        model.model.decoder.projection_configs = projection_configs
        for step, batch in enumerate(infer_dataloader):
            # TODO, add prompts, choosen, rejected
            # implementation, batch = {k: v.to(device) for k, v in batch.items()}
            sources_sequences += batch['sources']
            ground_truths += batch['gts']
            del batch['sources']
            del batch['gts']
            batch = to_device(batch, device)
            prompt_len = batch['input_ids'].shape[1]
            # update progress bar
            progress_bar.update(1)
            description = f"Step {step}"
            progress_bar.set_description(description, refresh=False)
            with torch.no_grad():
                # TODO, add more inference params
                # backbone config
                # generate_ids = model.generate(batch['input_ids'], max_new_tokens=args.max_ans_len,
                #                               pad_token_id=tokenizer.eos_token_id, attention_mask = batch['attention_mask'], temperature=0.7, do_sample=True, repetition_penalty=2.0 )
                # sft config
                generate_ids = model.generate(input_ids=batch['input_ids'],
                                              attention_mask=batch['attention_mask'],
                                              max_new_tokens=args.max_ans_len,
                                              bos_token_id=tokenizer.bos_token_id,
                                              eos_token_id=tokenizer.eos_token_id,
                                              pad_token_id=tokenizer.unk_token_id,
                                              temperature=args.temperature,
                                              do_sample=True,
                                              num_return_sequences=1,
                                              use_cache=True
                                              )
            sequences = tokenizer.batch_decode(generate_ids[:, prompt_len:], skip_special_tokens=True,
                                               clean_up_tokenization_spaces=False)
            predicted_sequences += sequences
        return sources_sequences, predicted_sequences, ground_truths

    def save_inference_results(evaluation_result: dict, sources_sequences: list, predicted_sequences: list,
                                ground_truths: list, round: int, i_task: int, task: str):
        # save as a json file
        df = {"eval": evaluation_result, 'prompts': sources_sequences, 'results': predicted_sequences,
                'labels': ground_truths}
        if not os.path.exists(args.inference_output_path):
            os.makedirs(args.inference_output_path)
        with open(args.inference_output_path + "/results-" + str(round) + "-" + str(i_task) + "-" + task + ".json", "w+", encoding='utf-8') as file:
            json.dump(df, file, ensure_ascii=False)


    tokenizer = load_hf_tokenizer(args.model_name_or_path, fast_tokenizer=True)
    # default the LLM is decoder only model, so padding side is left
    assert tokenizer.padding_side == 'left'
    assert tokenizer.truncation_side == "left"

    # set evaluation batch size
    # only support bs = 1, cause right padding training logic
    # TODO, modify left pad for training and inference
    inference_tasks = args.inference_tasks 
    task_num = len(inference_tasks)
    for round in range(task_num):   # load models and adapters of a new round in continual learning
        inference_model_path = os.path.join(args.inference_model_path, str(round))
        print_rank_0("Inference Model Path: " + inference_model_path, args.local_rank)

        if args.CL_method == 'SVD':
            if 'opt' in args.model_name_or_path:
                model = create_hf_model(CustomOPTForCausalLM,
                                        args.model_name_or_path,
                                        tokenizer,
                                        ds_config=None,
                                        args=args,
                                        )
            elif 'bloom' in args.model_name_or_path:
                model = create_hf_model(CustomBloomForCausalLM,
                                        args.model_name_or_path,
                                        tokenizer,
                                        ds_config=None,
                                        args=args,
                                        )
        else:
            model = create_hf_model(AutoModelForCausalLM,
                                    args.model_name_or_path,
                                    tokenizer,
                                    ds_config=None,
                                    args=args,
                                    )

        projection_configs = None
        PROJ_CONFIG_PATH = args.proj_config_path + args.model + '_' + str(args.repurpose_dim_size) + '_proj_config.pkl'
        with open(PROJ_CONFIG_PATH, 'rb') as f:
            projection_configs = pickle.load(f)
            print("projection configs has been loaded from disk.")

        print(device)
        # Assuming projection_configs is a tuple of lists of tensors
        if args.CL_method == 'SVD':
            projection_configs = tuple(
                [(a.to(device).to(torch.float32), b.to(device).to(torch.float32), c.to(device).to(torch.float32)) for a,b,c in config_list] for config_list in projection_configs
            )
        # TODO: add adapters
        if args.CL_method == "LFPT5":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, inference_model_path)

        if args.CL_method == "O-LoRA":
            from utils.my_peft import PeftModel
            model = PeftModel.from_pretrained(model, inference_model_path)
            for name, param in model.named_parameters():
                if name.find("loranew_") != -1:
                    param.requires_grad = True
                elif name.find("lora_") != -1:
                    param.requires_grad = False

        if args.CL_method == "OGD":
            from peft import get_peft_model, LoraConfig, TaskType
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=32, lora_dropout=0.1
            )
            model = get_peft_model(model, peft_config)
            for name, param in model.named_parameters():
                if name.find("lora") != -1:
                    param.requires_grad = True

        if args.CL_method=="PP" or args.CL_method=="L2P":
            if "opt" in args.model_name_or_path.lower():
                embed_tokens_shape = model.model.decoder.embed_tokens.weight.shape
                embed_tokens = model.model.decoder.embed_tokens
                args.embed_tokens_dim = embed_tokens_shape[1]
                args.embed_tokens_length = embed_tokens_shape[0]
                args.embed_tokens = embed_tokens
            elif "llama" in args.model_name_or_path.lower():
                embed_tokens_shape = model.model.embed_tokens.weight.shape
                embed_tokens = model.model.embed_tokens
                args.embed_tokens_dim = embed_tokens_shape[1]
                args.embed_tokens_length = embed_tokens_shape[0]
                args.embed_tokens = embed_tokens
            if args.CL_method=="PP":
                args.prefix_len = 20
                model = convert_PP_model(model, args)
            elif args.CL_method=="L2P":
                args.pool_size = 10
                args.prompt_length = 5
                args.prompt_init = "uniform"
                model = convert_L2P_model(model, args)
                for name, params in model.named_parameters():
                    if "prompt" not in name:
                        params.requires_grad=False

        if args.CL_method == "lora":
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, inference_model_path)

        if args.CL_method != "lora" and args.CL_method != "O-LoRA" and args.CL_method != "LFPT5": 
            inference_model = torch.load(os.path.join(inference_model_path, "pytorch_model.bin"))
            for name, param in model.named_parameters():
                param.data.copy_(inference_model[name])
            del inference_model

        model.to(device)

        for inference_task_id in range(round+1):    # evaluation for previous tasks in a single round
            inference_task = inference_tasks[inference_task_id]
            dataset_path = os.path.join(args.data_path, inference_task)
            # Prepare the data
            _, _, infer_dataset = create_prompt_dataset(
                args.local_rank,
                dataset_path,
                args.data_output_path,
                args.seed,
                distributed=False
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
            infer_sampler = SequentialSampler(infer_dataset)
            infer_dataloader = DataLoader(infer_dataset,
                                          collate_fn=inf_data_collator,
                                          sampler=infer_sampler,
                                          batch_size=args.inference_batch)
            progress_bar = tqdm(total=len(infer_dataloader), leave=True)

            # Inference !
            print_rank_0("***** Start inference *****", args.local_rank)
            sources_sequences, predicted_sequences, ground_truths = prediction(model, infer_dataloader, inference_task_id, projection_configs)
            # Get Accuracy/ROUGE/BLEU/...
            # The evaluation result is stored in a dictionary. e.g. {"accuracy": .., "rouge-L": ..}
            try:
                if inference_task == "ScienceQA":
                    evaluation_result = eval_ScienceQA.eval(predicted_sequences, ground_truths)
                elif inference_task == "MeetingBank":
                    evaluation_result = eval_MeetingBank.eval(predicted_sequences, ground_truths)
                elif inference_task == "C-STANCE":
                    evaluation_result = eval_CStance.eval(predicted_sequences, ground_truths)
                elif inference_task == "Papyrus-f":
                    evaluation_result = eval_PapyrusF.eval(predicted_sequences, ground_truths)
                elif inference_task == "Py150":
                    evaluation_result = eval_Py150.eval(predicted_sequences, ground_truths)
                elif inference_task == "FOMC":
                    evaluation_result = eval_FOMC.eval(predicted_sequences, ground_truths)
                elif inference_task == "NumGLUE-cm":
                    evaluation_result = eval_NumGLUE_cm.eval(predicted_sequences, ground_truths)
                elif inference_task == "NumGLUE-ds":
                    evaluation_result = eval_NumGLUE_ds.eval(predicted_sequences, ground_truths)
                elif inference_task == "20Minuten":
                    evaluation_result = eval_20Minuten.eval(sources_sequences, predicted_sequences, ground_truths)
                else:
                    evaluation_result = {}
            except Exception as e:
                print(e)

            # if args.global_rank <= 0:  # only one process is running
            print("***** Saving inference results *****")
            save_inference_results(evaluation_result, sources_sequences, predicted_sequences, ground_truths, round, inference_task_id, inference_task)

if __name__ == "__main__":
    main()
