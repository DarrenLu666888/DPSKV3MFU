# Args ref from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/configs/config_671B.json
# follow code from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py

from typing import Tuple, Optional, Literal
import json
from dataclasses import dataclass

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.
    # customized args
    attn_impl: str = 'absorb'

# class Args:
#     def __init__(self):
#         self.vocab_size = 129280
#         self.dim = 7168
#         self.inter_dim = 18432
#         self.moe_inter_dim = 2048
#         self.n_layers = 61
#         self.n_dense_layers = 3
#         self.n_heads = 128
#         self.n_routed_experts = 256
#         self.n_shared_experts = 1
#         self.n_activated_experts = 8
#         self.n_expert_groups = 8
#         self.n_limited_groups = 4
#         self.route_scale = 2.5
#         self.score_func = "sigmoid"
#         self.q_lora_rank = 1536
#         self.kv_lora_rank = 512
#         self.qk_nope_head_dim = 128
#         self.qk_rope_head_dim = 64
#         self.v_head_dim = 128
#         self.dtype = "fp8"
#         self.attn_impl = "absorb" # ["naive", "absorb"]

with open('configs/config_16B.json') as f:
    args = ModelArgs(**json.load(f))

# we assume the T, B, M in the paper are in the unit of 1000
BASE = 1000

def cal_embed_fwd_flops(bs: int, seq_len: int):
    # y = F.embedding(x, self.weight)
    return 2 * bs * seq_len * args.dim

def cal_head_fwd_flops(bs: int, seq_len: int):
    return 2 * bs * seq_len * args.dim * args.vocab_size

# def cal_attn_fwd_flops(bs: int, seq_len: int):
#     # score = Q x K^T /2 double to causal
#     # scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale -> [bs, seq_len, seq_len]
#     flops = 2 * bs * seq_len * seq_len * args.n_heads * args.qk_head_dim

#     # score x V
#     # x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos]) -> [bs, seq_len, args.n_heads, args.v_head_dim]
#     flops += 2 * bs * seq_len * seq_len * args.n_heads * args.v_head_dim

#     return flops / 2

def cal_mla_fwd_flops(bs: int, seq_len: int, cur_token_id: int):
    flops = 0

    args.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim

    # Q down + up
    # q = self.wq_b(self.q_norm(self.wq_a(x))) -> [bs, seq_len, (args.n_heads * args.qk_head_dim)]
    # 1. ignore rmsnorm : bs * seq_len * (args.q_lora_rank + args.q_lora_rank + 3 + args.q_lora_rank + 2)
    # 2. 2 * : mul and add
    flops += 2 * bs * seq_len * args.dim * args.q_lora_rank
    flops += 2 * bs * seq_len * args.q_lora_rank * args.n_heads * args.qk_head_dim

    # KV down
    # kv = self.wkv_a(x) -> [bs, seq_len, (args.kv_lora_rank + args.qk_rope_head_dim)]
    flops += 2 * bs * seq_len * args.dim * (args.kv_lora_rank + args.qk_rope_head_dim)

    if (args.attn_impl == 'naive'):
        # KV up
        # kv = self.wkv_b(self.kv_norm(kv)) -> [bs, seq_len, (args.n_heads * args.qk_head_dim)]
        # 1. ignore kv_norm
        flops += 2 * bs * seq_len * args.kv_lora_rank * args.n_heads * (args.qk_nope_head_dim + args.v_head_dim)

        # score
        # scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale -> [bs, seq_len, args.n_heads, cur_token_id]
        # 1. ignore softmax_scale : bs * seq_len * args.n_heads * cur_token_id
        # 2. ignore mask only in prefill: bs * seq_len * seq_len * args.n_heads
        # 3. ignore softmax : bs * seq_len * args.n_heads * (cur_token_id + cur_token_id-1 + cur_token_id)
        flops += 2 * bs * seq_len * cur_token_id * args.n_heads * args.qk_head_dim # for prefill: cur_token_id('t' in 'bthd')=seq_len,seq_len=input_len; for generate: cur_token_id=input_len+generate_len,seq_len=1
        # x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos]) -> [bs, seq_len, args.n_heads, args.v_head_dim]
        flops += 2 * bs * seq_len * args.n_heads * args.v_head_dim * cur_token_id
    else: # absorb
        # q k absorb
        # 1. ignore weight_dequant : 
        # q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim]) -> [bs, seq_len, args.n_heads, args.kv_lora_rank]
        flops += 2 * bs * seq_len * args.n_heads * args.kv_lora_rank * args.qk_nope_head_dim

        # score
        # scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
        #           torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
        # 1. ignore kv_norm
        # 2. ignore softmax_scale : bs * seq_len * args.n_heads * cur_token_id
        flops += 2 * bs * seq_len * args.n_heads * cur_token_id * args.kv_lora_rank
        flops += 2 * bs * seq_len * args.n_heads * cur_token_id * args.qk_rope_head_dim
        # x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])
        flops += 2 * bs * seq_len * args.n_heads * args.kv_lora_rank * cur_token_id
        # x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])
        flops += 2 * bs * seq_len * args.n_heads * args.v_head_dim * args.qk_rope_head_dim

    # x = self.wo(x.flatten(2))
    flops += 2 * bs * seq_len * args.n_heads * args.v_head_dim * args.dim

    return flops

def cal_moe_fwd_flops(bs: int, seq_len: int):

    flops = 0
    # 1. ignore gate : 
    # 2. ignore
    # 3. ignore ……
    # (in expert, the reason of *3) self.w2(F.silu(self.w1(x)) * self.w3(x))
    flops += 2 * bs * seq_len * args.dim * args.moe_inter_dim * 3
    # * (dot)
    flops += bs * seq_len * args.moe_inter_dim

    return flops

def cal_mlp_fwd_flops(bs: int, seq_len: int):
    flops = 2 * bs * seq_len * args.dim * args.inter_dim * 3
    # * (dot)
    flops += bs * seq_len * args.inter_dim
    return flops

def cal_fwd_flops(bs: int, seq_len: int, cur_token_id: int):
    """
        flops (TFLOPS) per token
    """

    flops_mla = cal_mla_fwd_flops(bs, seq_len, cur_token_id)  / (BASE**3) * args.n_layers
    flops_moe = (args.n_shared_experts + args.n_activated_experts) * cal_moe_fwd_flops(bs, seq_len)  / (BASE**3) * (args.n_layers - args.n_dense_layers)
    flops_mlp = cal_mlp_fwd_flops(bs, seq_len)   / (BASE**3) * args.n_dense_layers


    flops_embed = cal_embed_fwd_flops(bs, seq_len)   / (BASE**3)
    flops_head = cal_head_fwd_flops(bs, seq_len)   / (BASE**3)

    # print(f"flops_mla: {flops_mla} TFLOPS, flops_moe: {flops_moe} TFLOPS")

    flops = flops_mla + flops_moe + flops_mlp + flops_embed + flops_head
    
    # print(f"flops in bs({bs})seq_len: {seq_len}, flops: {flops} TFLOPS")
    return flops



# pre-training context length 4K
# seq_len = 1024 * 4

# The following five data depend on the specific conditions of the test set and the inference framework + running device.
bsz = 2
H100_peak_bf16_flops = 989.5 * 1e12 / BASE**4 # TFLOPS
gpu_hours = 2.664 * 3600 # seconds
input_tokens = 128 
new_tokens_generated = 512

total_flops = 0
# prefill
total_flops += cal_fwd_flops(bsz, input_tokens, cur_token_id=input_tokens)
print('prefill flops: {:.2f} TFLOPS', total_flops)
# decode
for i in range(new_tokens_generated):
    total_flops += cal_fwd_flops(bsz, 1, cur_token_id=i+input_tokens+1)

# bwd_flops = fwd_flops * 2 # in inference, no bwd

MFU = 1000*(total_flops / BASE) / (gpu_hours  * H100_peak_bf16_flops)

print(f"we assume the T, B, M in the paper are in the unit of {BASE}")
print(f"MFU: {MFU}")

# # estimate MFU from parameter numbers
# attn_flosp = 3 * cal_attn_fwd_flops(bsz, seq_len) * args.n_layers / (BASE**3) / (bsz * seq_len)
# MFU_ref = (37*6 + attn_flosp) * 14.8 / (gpu_hours * H100_peak_bf16_flops)
# print(f"ref MFU: {MFU_ref}")
