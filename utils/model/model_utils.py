# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import random
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import LlamaForCausalLM, LlamaConfig

import transformers

def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    args=None,
                    ):
    transformers.logging.set_verbosity_info()
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    print(model_config)
    #print(model_config)
    if args is not None:
        print('we assign the flags and step size etc....')
        if args.CL_method == 'SVD' or args.CL_method == 'svd_replay' or args.CL_method == 'svd_replay_alignment' or args.CL_method == 'svd_alignment':
            model_config.use_repurposed_dims = args.use_repurposed_dims
            print('foooooooooooooo')
            print(args.ffn_only)
            print(args.mha_only)
            model_config.ffn_only = args.ffn_only
            model_config.mha_only = args.mha_only
            model_config.qk_only = args.qk_only
            model_config.ov_only = args.ov_only
            model_config.step_size = args.step_size
            model_config.project_only_first_layer = args.project_only_first_layer
            #model_config.repurpose_direction_list = args.repurpose_direction_list
            print(model_config)

    
    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=True,
        resume_download=True)

    # TODO: generalize this with other models like GPT
    if 'llama' in model_name_or_path or 'vicuna' in model_name_or_path:
        print('end token -----> eos token')
        # llama use eos_token_id but not end_token_id
        model.config.end_token_id = tokenizer.eos_token_id
    if 'llama' in model_name_or_path or 'vicuna' in model_name_or_path or 'opt' in model_name_or_path:
        # compatible with OPT and llama2
        print('pad token ====== eos token')
        model.config.pad_token_id = model.config.eos_token_id
    
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


def create_hf_model_analysis(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    args=None,
                    ):
    transformers.logging.set_verbosity_info()
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    print(model_config)
    #print(model_config)
    
    (use_repurposed_dims, ffn_only, mha_only, qk_only, ov_only, step_size, project_only_first_layer) = args
        
    model_config.use_repurposed_dims = use_repurposed_dims
    model_config.ffn_only = ffn_only
    model_config.mha_only = mha_only
    model_config.qk_only = qk_only
    model_config.ov_only = ov_only
    model_config.step_size = step_size
    model_config.project_only_first_layer = project_only_first_layer

    print(model_config)

    if disable_dropout:
        model_config.dropout = 0.0
    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    model = model_class.from_pretrained(
        model_name_or_path,
        from_tf=bool(".ckpt" in model_name_or_path),
        config=model_config,
        trust_remote_code=True,
        resume_download=True)

    # TODO: generalize this with other models like GPT
    if 'llama' in model_name_or_path or 'vicuna' in model_name_or_path:
        print('end token -----> eos token')
        # llama use eos_token_id but not end_token_id
        model.config.end_token_id = tokenizer.eos_token_id
    if 'llama' in model_name_or_path or 'vicuna' in model_name_or_path or 'opt' in model_name_or_path:
        # compatible with OPT and llama2
        print('pad token ====== eos token')
        model.config.pad_token_id = model.config.eos_token_id
    
    model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    return model


def get_latent_directions_module(module):
    # TODO: this does not support mixed precision
    # RuntimeError: "svd_cuda_gesvdj" not implemented for 'BFloat16'
    # 
    #print(layer)
    # TODO: implement mixed precision check
    #tensor_float = layer.weight.data.to(torch.float32)
    U, S, VH = torch.linalg.svd(module)
    VH = VH.to(torch.bfloat16)
    V = VH.mH
    return V

def get_latent_directions(model):
    results = []

    # Assume 'first_layer_name' is the name pattern that identifies weights in the first layer
    first_layer_name = "model.layers.0"  # replace with actual pattern

    for name, param in model.named_parameters():
        if ("weight" in name) and (len(param.size()) == 2) and (first_layer_name in name):
            print(f'Processing {name}...')

            U, S, VH = torch.linalg.svd(param.data)
            significant_singular_values = sum(s > 1e-5 for s in S)
            print(f"{name}: {significant_singular_values} significant singular values out of {len(S)}")

            #results.append((name, U, S, VH))
            results.append((name, VH))

    return results



def project_to_subspaces(input_tensor: torch.Tensor, basis: torch.tensor, 
                         repurposed_directions: torch.tensor, base_directions: torch.tensor,
                         step_size=None, use_repurposed_dims=True, i_task=0, repurpose_direction_list=None):
    """
    Project each element in the sequence of input_tensor on the base subspace,
    then traverse the projected element along the repurposed directions.
    
    Args:
        input_tensor (torch.Tensor): Tensor of shape [batch_size, sequence_length, hidden_size].
        basis (torch.Tensor): Basis vectors for projection.
        repurposed_dims (torch.Tensor): Dimensions for traversal.
        base_dims (torch.Tensor, optional): Dimensions for base subspace. If None, uses all non-repurposed dims.
        step_size (float or torch.Tensor, optional): Step sizes for traversal.

    Returns:
        torch.Tensor: The tensor after projection and traversal.
    """
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(dim=0)
    batch_size, sequence_length, hidden_size = input_tensor.shape

    #if base_dims is None:
        # Take all non-repurposed dims to span the base subspace -- default mode
    #    base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])

    # Use values instead of boolean to change order as needed
    #print('basis shape', basis.shape)
    #print('repurposed dimensions', repurposed_dims)
    #print('repurposed dimensions shape', repurposed_dims.shape)
    
    # TODO: check if the components/axes are correct
    #repurposed_directions = basis[:, repurposed_dims].to(input_tensor.device)
    #base_directions = basis[:, base_dims].to(input_tensor.device)

    # Reshape input_tensor to merge batch and sequence dimensions
    reshaped_tensor = input_tensor.view(-1, hidden_size)

    # Project the reshaped tensor
    #if use_repurposed_dims:
        # project to dormant subspace
    #    projected_tensor = reshaped_tensor @ repurposed_directions
    #    base_tensor = projected_tensor @ repurposed_directions.T
    #else:
        # project to strong/dominant/base subspace

    
    projected_tensor = reshaped_tensor @ base_directions
    base_tensor = projected_tensor @ base_directions.T
    
    # base_directions : VV^T
    #base_tensor = reshaped_tensor @ base_directions
    if step_size is None:
        # Reshape back to original dimensions and return
        return base_tensor.view(batch_size, sequence_length, -1)

    if isinstance(step_size, float) or isinstance(step_size, int):
        step_size = torch.tensor([step_size]).to(input_tensor.device)
    
    #print('the current task index is ', i_task)
    #random.seed(i_task)
    #direction_idx = repurpose_direction_list[i_task]
    #random_direction = random.randint(0, repurposed_directions.shape[1]-1)
    #print('the random direction to repurpose is', i_task)
    repurposed_direction = repurposed_directions[:, i_task]
    #print('repurposed_direction', repurposed_direction.shape)
    #repurposed_direction = repurposed_direction.T
    #print('repurposed_direction.T', repurposed_direction.shape)
    #print('step size', step_size.shape)
    if step_size.dim() == 1:
        # separate same-sized steps on all dims
        #num_steps = step_size.shape[0]
        #edits = torch.einsum('a, df -> adf', step_size, repurposed_direction).to(torch.bfloat16)
        #edits = edits.squeeze(0)
        edits = step_size * repurposed_direction
    else:
        raise NotImplementedError('Cannot edit with these values')

    #elif step_size.dim() == 3:
        # TODO: maybe remove this
        # compound steps, on multiple dims
    #    edits = step_size @ repurposed_direction

    edit_tensors = base_tensor + edits
    #print('edit tensor shape', edit_tensors.shape)
    #edit_tensors = edit_tensors.sum(dim=1)
    #print('sum edit tensor shape', edit_tensors.shape)
    #print(edit_tensors.dtype)
    #print(batch_size, sequence_length, *edit_tensors.shape[1:])
    #edit_tensors = edit_tensors.view(-1, hidden_size)  # Flatten for reshaping
    # Reshape back to [batch_size, sequence_length, hidden_size, ...]
    return edit_tensors.view(batch_size, sequence_length, hidden_size)


def project_to_subspaces_bloom_qkv(input_tensor: torch.Tensor, basis: torch.tensor, 
                         repurposed_directions: torch.tensor, base_directions: torch.tensor,
                         step_size=None, use_repurposed_dims=True, i_task=0):
    
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(dim=0)

    batch_size, sequence_length, hidden_size = input_tensor.shape
    _, base_3_hidden_size = base_directions.shape
    _, repurposed_3_hidden_size = repurposed_directions.shape
    #print('base 3hidden size', base_3_hidden_size)
    #print('repurposed_3_hidden_size', repurposed_3_hidden_size)
    base_hidden_size = base_3_hidden_size // 3
    repurposed_hidden_size = repurposed_3_hidden_size // 3
    #print('base_hidden_size', base_hidden_size)
    #print('repurposed_hidden_size', repurposed_hidden_size)
    #print('input tensor shape', input_tensor.shape)
    reshaped_tensor = input_tensor.view(-1, hidden_size)
    #print('reshaped tensor shape', reshaped_tensor.shape)
    #print('base direction shape', base_directions.shape)
    #print('repurposed directions shape', repurposed_directions.shape)
    q_base_directions = base_directions[:, :base_hidden_size]
    q_repurposed_directions = repurposed_directions[:, :repurposed_hidden_size]

    k_base_directions = base_directions[:, base_hidden_size:2*base_hidden_size]
    k_repurposed_directions = repurposed_directions[:, repurposed_hidden_size:2*repurposed_hidden_size]

    v_base_directions = base_directions[:, 2*base_hidden_size:3*base_hidden_size]
    v_repurposed_directions = repurposed_directions[:, 2*repurposed_hidden_size:3*repurposed_hidden_size]
    
    q_projected_tensor = reshaped_tensor @ q_base_directions
    q_base_tensor = q_projected_tensor @ q_base_directions.T
    
    k_projected_tensor = reshaped_tensor @ k_base_directions
    k_base_tensor = k_projected_tensor @ k_base_directions.T
    
    v_projected_tensor = reshaped_tensor @ v_base_directions
    v_base_tensor = v_projected_tensor @ v_base_directions.T
    
    #print('base tensor shapes after projection', q_base_tensor.shape, k_base_tensor.shape, v_base_tensor.shape)
    if step_size is None:
        # Reshape back to original dimensions and return
        #print('qkv base tensor shapes', q_base_tensor.shape, k_base_tensor.shape, v_base_tensor.shape)
        base_tensor = torch.cat((q_base_tensor, k_base_tensor, v_base_tensor), 1)
        #print('base tensor shape', base_tensor.shape)
        return base_tensor.view(batch_size, sequence_length, -1)

    if isinstance(step_size, float) or isinstance(step_size, int):
        step_size = torch.tensor([step_size]).to(input_tensor.device)
    
    #random.seed(i_task)
    #print(q_repurposed_directions.shape)
    #print(k_repurposed_directions.shape)
    #print(v_repurposed_directions.shape)
    #q_random_direction = random.randint(0, q_repurposed_directions.shape[1]-1)
    #k_random_direction = random.randint(0, k_repurposed_directions.shape[1]-1)
    #v_random_direction = random.randint(0, v_repurposed_directions.shape[1]-1)

    q_repurposed_direction = q_repurposed_directions[:, i_task]
    k_repurposed_direction = k_repurposed_directions[:, i_task]
    v_repurposed_direction = v_repurposed_directions[:, i_task]

    if step_size.dim() == 1:
        q_edits = step_size * q_repurposed_direction
        k_edits = step_size * k_repurposed_direction
        v_edits = step_size * v_repurposed_direction
    else:
        raise NotImplementedError('Cannot edit with these values')

    q_edit_tensors = q_base_tensor + q_edits
    k_edit_tensors = k_base_tensor + k_edits
    v_edit_tensors = v_base_tensor + v_edits
    #print('edit tensor shapes', q_edit_tensors.shape, k_edit_tensors.shape, v_edit_tensors.shape)
    #edit_tensors = torch.cat((q_edit_tensors, k_edit_tensors, v_edit_tensors), 1)
    #print('edit tensor shape', edit_tensors.shape)

    #return edit_tensors.view(batch_size, sequence_length, 3*hidden_size)
    return q_edit_tensors.view(batch_size, sequence_length, hidden_size), k_edit_tensors.view(batch_size, sequence_length, hidden_size), v_edit_tensors.view(batch_size, sequence_length, hidden_size)




def project_to_subspaces_gpt2_c_attn(input_tensor: torch.Tensor, basis: torch.tensor, 
                         repurposed_directions: torch.tensor, base_directions: torch.tensor,
                         step_size=None, use_repurposed_dims=True, i_task=0):
    
    if input_tensor.dim() == 2:
        input_tensor = input_tensor.unsqueeze(dim=0)

    batch_size, sequence_length, hidden_size = input_tensor.shape

    _, base_3_hidden_size = base_directions.shape

    _, repurposed_3_hidden_size = repurposed_directions.shape
    base_hidden_size = base_3_hidden_size // 3
    repurposed_hidden_size = repurposed_3_hidden_size // 3
    reshaped_tensor = input_tensor.view(-1, hidden_size)
    q_base_directions = base_directions[:, :base_hidden_size]
    q_repurposed_directions = repurposed_directions[:, :repurposed_hidden_size]

    k_base_directions = base_directions[:, base_hidden_size:2*base_hidden_size]
    k_repurposed_directions = repurposed_directions[:, repurposed_hidden_size:2*repurposed_hidden_size]

    v_base_directions = base_directions[:, 2*base_hidden_size:3*base_hidden_size]
    v_repurposed_directions = repurposed_directions[:, 2*repurposed_hidden_size:3*repurposed_hidden_size]
    
    q_projected_tensor = reshaped_tensor @ q_base_directions
    q_base_tensor = q_projected_tensor @ q_base_directions.T
    
    k_projected_tensor = reshaped_tensor @ k_base_directions
    k_base_tensor = k_projected_tensor @ k_base_directions.T
    
    v_projected_tensor = reshaped_tensor @ v_base_directions
    v_base_tensor = v_projected_tensor @ v_base_directions.T
    
    #print('base tensor shapes after projection', q_base_tensor.shape, k_base_tensor.shape, v_base_tensor.shape)
    if step_size is None:
        # Reshape back to original dimensions and return
        #print('qkv base tensor shapes', q_base_tensor.shape, k_base_tensor.shape, v_base_tensor.shape)
        base_tensor = torch.cat((q_base_tensor, k_base_tensor, v_base_tensor), 1)
        #print('base tensor shape', base_tensor.shape)
        return base_tensor.view(batch_size, sequence_length, -1)

    if isinstance(step_size, float) or isinstance(step_size, int):
        step_size = torch.tensor([step_size]).to(input_tensor.device)
    
    #random.seed(i_task)
    #print(q_repurposed_directions.shape)
    #print(k_repurposed_directions.shape)
    #print(v_repurposed_directions.shape)
    #q_random_direction = random.randint(0, q_repurposed_directions.shape[1]-1)
    #k_random_direction = random.randint(0, k_repurposed_directions.shape[1]-1)
    #v_random_direction = random.randint(0, v_repurposed_directions.shape[1]-1)

    q_repurposed_direction = q_repurposed_directions[:, i_task]
    k_repurposed_direction = k_repurposed_directions[:, i_task]
    v_repurposed_direction = v_repurposed_directions[:, i_task]

    if step_size.dim() == 1:
        q_edits = step_size * q_repurposed_direction
        k_edits = step_size * k_repurposed_direction
        v_edits = step_size * v_repurposed_direction
    else:
        raise NotImplementedError('Cannot edit with these values')

    q_edit_tensors = q_base_tensor + q_edits
    k_edit_tensors = k_base_tensor + k_edits
    v_edit_tensors = v_base_tensor + v_edits
    #print('edit tensor shapes', q_edit_tensors.shape, k_edit_tensors.shape, v_edit_tensors.shape)
    #edit_tensors = torch.cat((q_edit_tensors, k_edit_tensors, v_edit_tensors), 1)
    #print('edit tensor shape', edit_tensors.shape)

    #return edit_tensors.view(batch_size, sequence_length, 3*hidden_size)
    return q_edit_tensors.view(batch_size, sequence_length, hidden_size), k_edit_tensors.view(batch_size, sequence_length, hidden_size), v_edit_tensors.view(batch_size, sequence_length, hidden_size)



def projection_pipeline(input_tensor, layer):
    basis = get_latent_directions_module(layer)
    repurposed_dims_size = 100
    batch_size, sequence_length, hidden_size = input_tensor.shape

    assert hidden_size >= repurposed_dims_size
    repurposed_dims = torch.arange(hidden_size - repurposed_dims_size, hidden_size)
    #print('projecting!!!!')

    return project_to_subspaces(input_tensor, basis, repurposed_dims)


def generate_basis_pipeline(model, repurposed_dims_size):
    gate_proj_bases = []
    up_proj_bases = []
    repurposed_dims_list = []
    for name, param in model.model.layers.named_parameters():
        if 'gate_proj' in name:
            print(name)
            print(param.shape)
            hidden_size = param.shape[1]
            gate_proj_bases.append((get_latent_directions_module(param), torch.arange(hidden_size - repurposed_dims_size, hidden_size)))
            #repurposed_dims_list.append(torch.arange(hidden_size - repurposed_dims_size, hidden_size))

        elif 'up_proj' in name:
            print(name)
            print(param.shape)
            hidden_size = param.shape[1]
            up_proj_bases.append((get_latent_directions_module(param), torch.arange(hidden_size - repurposed_dims_size, hidden_size)))
            #repurposed_dims_list.append(torch.arange(hidden_size - repurposed_dims_size, hidden_size))

    return (gate_proj_bases, up_proj_bases)

def generate_basis_for_opt(model, repurpose_dim_size):
    k_proj_bases = []
    v_proj_bases = []
    q_proj_bases = []
    out_proj_bases = []
    #self_attn_layer_norm_bases = []
    fc1_bases = []
    fc2_bases = []
    
    repurposed_dims_list = []
    for name, param in model.model.decoder.layers.named_parameters():
        print(name, param)
        param = param - param.mean(dim=0)
        #torch.cuda.empty_cache()
        #param = param.to('cuda')
        if 'k_proj.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            k_proj_bases.append((basis, repurposed_directions, base_directions))
        elif 'v_proj.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            v_proj_bases.append((basis, repurposed_directions, base_directions))
        elif 'q_proj.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            q_proj_bases.append((basis, repurposed_directions, base_directions))
        elif 'out_proj.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            out_proj_bases.append((basis, repurposed_directions, base_directions))
        elif 'fc1.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            fc1_bases.append((basis, repurposed_directions, base_directions))
        elif 'fc2.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            fc2_bases.append((basis, repurposed_directions, base_directions))
        else:
            continue

    return (k_proj_bases, v_proj_bases, q_proj_bases, out_proj_bases, fc1_bases, fc2_bases)


def generate_basis_for_bloom(model, repurpose_dim_size):
    query_key_value_bases = []
    dense_bases = []
    dense_h_to_4h_bases = []
    dense_4h_to_h_bases = []

    repurposed_dims_list = []

    for name, param in model.transformer.h.named_parameters():
        print(name, param)
        torch.cuda.empty_cache()
        #param = param.to('cuda')
        if 'query_key_value.weight' in name:
            print(name, param)
            print(param.shape)
            hidden_size = param.shape[1] 
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])

            q_layer = param[:hidden_size, :]
            k_layer = param[hidden_size:2*hidden_size, :]
            v_layer = param[2*hidden_size:3*hidden_size, :]

            print(q_layer.shape, k_layer.shape, v_layer.shape)
            
            q_basis = get_latent_directions_module(q_layer)
            q_repurposed_directions = q_basis[:, repurposed_dims]
            q_base_directions = q_basis[:, base_dims]

            k_basis = get_latent_directions_module(k_layer)
            k_repurposed_directions = k_basis[:, repurposed_dims]
            k_base_directions = k_basis[:, base_dims]       

            v_basis = get_latent_directions_module(v_layer)
            v_repurposed_directions = v_basis[:, repurposed_dims]
            v_base_directions = v_basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            basis = torch.cat((q_basis, k_basis, v_basis), 1)
            repurposed_directions = torch.cat((q_repurposed_directions, k_repurposed_directions, v_repurposed_directions), 1)
            base_directions = torch.cat((q_base_directions, k_base_directions, v_base_directions), 1)
            query_key_value_bases.append((basis, repurposed_directions, base_directions))
        elif 'dense.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]  
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            #v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            dense_bases.append((basis, repurposed_directions, base_directions))          
        elif 'dense_h_to_4h.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            #v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            dense_h_to_4h_bases.append((basis, repurposed_directions, base_directions))
        elif 'dense_4h_to_h.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            #v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            dense_4h_to_h_bases.append((basis, repurposed_directions, base_directions))
    return (query_key_value_bases, dense_bases, dense_h_to_4h_bases, dense_4h_to_h_bases)




def generate_basis_for_gpt2(model, repurpose_dim_size):
    c_attn_bases = []
    c_proj_bases = []
    
    repurposed_dims_list = []

    for name, param in model.transformer.h.named_parameters():
        print(name, param)
        torch.cuda.empty_cache()
        #param = param.to('cuda')
        #TODO: implement FFN components
        if 'attn.c_attn.weight' in name:
            print(name, param)
            print(param.shape)
            param = param.T
            hidden_size = param.shape[1] 
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])

            q_layer = param[:hidden_size, :]
            k_layer = param[hidden_size:2*hidden_size, :]
            v_layer = param[2*hidden_size:3*hidden_size, :]

            print(q_layer.shape, k_layer.shape, v_layer.shape)
            
            q_basis = get_latent_directions_module(q_layer)
            q_repurposed_directions = q_basis[:, repurposed_dims]
            q_base_directions = q_basis[:, base_dims]

            k_basis = get_latent_directions_module(k_layer)
            k_repurposed_directions = k_basis[:, repurposed_dims]
            k_base_directions = k_basis[:, base_dims]       

            v_basis = get_latent_directions_module(v_layer)
            v_repurposed_directions = v_basis[:, repurposed_dims]
            v_base_directions = v_basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            basis = torch.cat((q_basis, k_basis, v_basis), 1)
            repurposed_directions = torch.cat((q_repurposed_directions, k_repurposed_directions, v_repurposed_directions), 1)
            base_directions = torch.cat((q_base_directions, k_base_directions, v_base_directions), 1)
            c_attn_bases.append((basis, repurposed_directions, base_directions))
        elif 'attn.c_proj.weight' in name:
            print(name, param)
            param = param.T
            hidden_size = param.shape[1]  
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            print('c proj basis ', basis.shape)
            #v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            c_proj_bases.append((basis, repurposed_directions, base_directions))          
        
    return (c_attn_bases, c_proj_bases)


def generate_basis_for_phi(model, repurpose_dim_size):
    query_key_value_bases = []
    dense_bases = []
    fc1_bases = []
    fc2_bases = []

    repurposed_dims_list = []

    for name, param in model.model.layers.named_parameters():
        print(name, param)
        torch.cuda.empty_cache()
        #param = param.to('cuda')
        if 'query_key_value.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            query_key_value_bases.append((basis, repurposed_directions, base_directions))
        elif 'dense.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]  
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            dense_bases.append((basis, repurposed_directions, base_directions))          
        elif 'fc1.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            fc1_bases.append((basis, repurposed_directions, base_directions))
        elif 'fc2.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            repurposed_dims = torch.arange(hidden_size - repurpose_dim_size, hidden_size)
            base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])
            basis = get_latent_directions_module(param)
            v = basis[:, base_dims]
            repurposed_directions = basis[:, repurposed_dims]
            base_directions = basis[:, base_dims]
            #v = v.to('cuda')
            #base_directions = v @ v.T
            #base_directions = base_directions.to('cpu')
            fc2_bases.append((basis, repurposed_directions, base_directions))
    return (query_key_value_bases, dense_bases, fc1_bases, fc2_bases)