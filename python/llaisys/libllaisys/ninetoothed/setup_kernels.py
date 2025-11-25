from kernels import add as add_kernel
from kernels import argmax as argmax_kernel
from kernels import embedding as embedding_kernel
from kernels import linear as linear_kernel
from kernels import rearrange as rearrange_kernel
from kernels import rms_norm as rms_norm_kernel    
from kernels import rope as rope_kernel
from kernels import self_attention as self_attention_kernel
from kernels import swiglu as swiglu_kernel

# import torch

# add examples:
def llaisysAdd(input, other, output):
    
    ## You need to convert input llaisys tensor into torch

    add_kernel.kernel(input, other, output, BLOCK_SIZE=1024)

    return output

# implement other operators below

