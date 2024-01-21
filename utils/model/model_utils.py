# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from huggingface_hub import snapshot_download
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import LlamaForCausalLM, LlamaConfig

def create_hf_model(model_class,
                    model_name_or_path,
                    tokenizer,
                    ds_config=None,
                    disable_dropout=False,
                    ):
    model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
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



def project_to_subspaces(input_tensor: torch.Tensor, basis: torch.Tensor,
                         repurposed_dims: torch.Tensor, base_dims: torch.Tensor = None,
                         step_size=None):
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

    if base_dims is None:
        # Take all non-repurposed dims to span the base subspace -- default mode
        base_dims = torch.tensor([x for x in range(hidden_size) if x not in repurposed_dims])

    # Use values instead of boolean to change order as needed
    #print('basis shape', basis.shape)
    #print('repurposed dimensions', repurposed_dims)
    #print('repurposed dimensions shape', repurposed_dims.shape)
    
    repurposed_directions = basis[:, repurposed_dims].to(input_tensor.device)
    base_directions = basis[:, base_dims].to(input_tensor.device)

    # Reshape input_tensor to merge batch and sequence dimensions
    reshaped_tensor = input_tensor.view(-1, hidden_size)

    # Project the reshaped tensor
    projected_tensor = reshaped_tensor @ base_directions
    base_tensor = projected_tensor @ base_directions.T

    if step_size is None:
        # Reshape back to original dimensions and return
        return base_tensor.view(batch_size, sequence_length, -1)

    if isinstance(step_size, float) or isinstance(step_size, int):
        step_size = torch.tensor([step_size]).to(input_tensor.device)

    repurposed_directions = repurposed_directions.T

    if step_size.dim() == 1:
        # separate same-sized steps on all dims
        num_steps = step_size.shape[0]
        edits = torch.einsum('a, df -> adf', step_size, repurposed_directions)
    elif step_size.dim() == 3:
        # compound steps, on multiple dims
        edits = step_size @ repurposed_directions
    else:
        raise NotImplementedError('Cannot edit with these values')

    edit_tensors = base_tensor.unsqueeze(1) + edits.unsqueeze(0).unsqueeze(-1)
    edit_tensors = edit_tensors.view(-1, hidden_size)  # Flatten for reshaping

    # Reshape back to [batch_size, sequence_length, hidden_size, ...]
    return edit_tensors.view(batch_size, sequence_length, *edit_tensors.shape[1:])

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

def generate_basis_for_opt(model, repurposed_dims_size):
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
        if 'k_proj.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            k_proj_bases.append((get_latent_directions_module(param), torch.arange(hidden_size - repurposed_dims_size, hidden_size)))
        elif 'v_proj.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            v_proj_bases.append((get_latent_directions_module(param), torch.arange(hidden_size - repurposed_dims_size, hidden_size)))
        elif 'q_proj.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            q_proj_bases.append((get_latent_directions_module(param), torch.arange(hidden_size - repurposed_dims_size, hidden_size)))
        elif 'out_proj.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            out_proj_bases.append((get_latent_directions_module(param), torch.arange(hidden_size - repurposed_dims_size, hidden_size)))
        elif 'fc1.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            fc1_bases.append((get_latent_directions_module(param), torch.arange(hidden_size - repurposed_dims_size, hidden_size)))
        elif 'fc2.weight' in name:
            print(name, param)
            hidden_size = param.shape[1]
            fc2_bases.append((get_latent_directions_module(param), torch.arange(hidden_size - repurposed_dims_size, hidden_size)))
        else:
            continue

    return (k_proj_bases, v_proj_bases, q_proj_bases, out_proj_bases, fc1_bases, fc2_bases)