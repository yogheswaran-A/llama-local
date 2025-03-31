import math
import time
import torch
import torch.nn as nn
from typing import Optional
from dataclasses import dataclass
from safetensors import safe_open
from safetensors.torch import load_file, save_file 
from torch.nn import functional as F
from transformers import AutoTokenizer

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 500000.0):
    # As written in the paragraph 3.2.2 of the paper
    # >> In order to generalize our results in 2D to any xi ∈ Rd where **d is even**, [...]
    assert head_dim % 2 == 0, "Dimension must be divisible by 2"
    # Build the theta parameter
    # According to the formula theta_i = 10000^(-2(i-1)/dim) for i = [1, 2, ... dim/2]
    # Shape: (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape: (Head_Dim / 2)
    theta = (theta ** (theta_numerator / head_dim)).to(device) # (Dim / 2)
    theta = theta**-1
    # Construct the positions (the "m" parameter)
    # Shape: (Seq_Len)
    m = torch.arange(seq_len, device=device)
    # Multiply each theta by each position using the outer product.
    # Shape: (Seq_Len) outer_product* (Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    # We can compute complex numbers in the polar form c = R * exp (i* (m * theta)), where R = 1 as follows:
    # (Seq_Len, Head_Dim / 2) -> (Seq_Len, Head_Dim / 2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex

def apply_rotary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device: str):
    # Separate the last dimension pairs of two values, representing the real and imaginary parts of the complex number
    # Two consecutive values will become a single complex number
    # (B, Seq_Len, H, Head_Dim) -> (B, Seq_Len, H, Head_Dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # Reshape the freqs_complex tensor to match the shape of the x_complex tensor. So we need to add the batch dimension and the head dimension
    # (Seq_Len, Head_Dim/2) --> (1, Seq_Len, 1, Head_Dim/2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # Multiply each complex number in the x_complex tensor by the corresponding complex number in the freqs_complex tensor
    # Which results in the rotation of the complex number as shown in the Figure 1 of the paper
    # (B, Seq_Len, H, Head_Dim/2) * (1, Seq_Len, 1, Head_Dim/2) = (B, Seq_Len, H, Head_Dim/2)
    x_rotated = x_complex * freqs_complex
    # Convert the complex number back to the real number
    # (B, Seq_Len, H, Head_Dim/2) -> (B, Seq_Len, H, Head_Dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_Len, H, Head_Dim/2, 2) -> (B, Seq_Len, H, Head_Dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        # (B, Seq_Len, N_KV_Heads, 1, Head_Dim)
        x[:, :, :, None, :]
        # (B, Seq_Len, N_KV_Heads, N_Rep, Head_Dim)
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        # (B, Seq_Len, N_KV_Heads * N_Rep, Head_Dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )

#from config.josn file 
@dataclass
class LlamaConfig:
    dim: int = 4096
    max_position_embeddings: int = 8192 # max_seq_len
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128256
    multiple_of: int = 1024
    ffn_dim_multiplier: float = 1.3
    intermediate_size: int = 14336
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0

    #
    device: str  = "cpu"
    max_batch_size: int = 32
    model_dir: str = "C:/yoghes/llms/Llama-3.1-8B-Instruct/"

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.down_proj = nn.Linear(config.intermediate_size,config.dim,bias = False)
        self.gate_proj = nn.Linear(config.dim,config.intermediate_size,bias = False)
        self.up_proj   = nn.Linear(config.dim,config.intermediate_size,bias = False)
        
    def forward(self, x: torch.Tensor):
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        swish = F.silu(self.gate_proj(x))
        # (B, Seq_Len, Dim) --> (B, Seq_Len, Hidden_Dim)
        x_V = self.up_proj(x)
        # (B, Seq_Len, Hidden_Dim) * (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Hidden_Dim)
        x = swish * x_V
        # (B, Seq_Len, Hidden_Dim) --> (B, Seq_Len, Dim)
        x = self.down_proj(x)
        return x

class CausalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_kv_heads = config.n_kv_heads
        # Indicates the number of heads for the Queries
        self.n_heads_q = config.n_heads
        self.n_rep = config.n_heads // config.n_kv_heads
        self.head_dim = config.dim // config.n_heads

        self.k_proj = nn.Linear(config.dim, self.head_dim * config.n_kv_heads, bias = False)
        self.o_proj = nn.Linear(config.dim, self.head_dim * config.n_heads, bias = False)
        self.q_proj = nn.Linear(self.head_dim * config.n_heads, config.dim, bias = False)
        self.v_proj = nn.Linear(config.dim, self.head_dim * config.n_kv_heads, bias = False)

        self.cache_k = torch.zeros((config.max_batch_size, config.max_position_embeddings, config.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((config.max_batch_size, config.max_position_embeddings, config.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape  # (B, 1, Dim)

        # (B, 1, Dim) -> (B, 1, H_Q * Head_Dim)
        xq = self.q_proj(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xk = self.k_proj(x)
        # (B, 1, Dim) -> (B, 1, H_KV * Head_Dim)
        xv = self.v_proj(x)

        # (B, 1, H_Q * Head_Dim) -> (B, 1, H_Q, Head_Dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_KV * Head_Dim) -> (B, 1, H_KV, Head_Dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # (B, 1, H_Q, Head_Dim) --> (B, 1, H_Q, Head_Dim)
        xq = apply_rotary_embeddings(xq, freqs_complex, device=x.device)
        # (B, 1, H_KV, Head_Dim) --> (B, 1, H_KV, Head_Dim)
        xk = apply_rotary_embeddings(xk, freqs_complex, device=x.device)

        # Replace the entry in the cache
        self.cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seq_len] = xv

        # (B, Seq_Len_KV, H_KV, Head_Dim)
        keys = self.cache_k[:batch_size, : start_pos + seq_len]
        # (B, Seq_Len_KV, H_KV, Head_Dim)
        values = self.cache_v[:batch_size, : start_pos + seq_len]

        # Since every group of Q shares the same K and V heads, just repeat the K and V heads for every Q in the same group.

        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        keys = repeat_kv(keys, self.n_rep)
        # (B, Seq_Len_KV, H_KV, Head_Dim) --> (B, Seq_Len_KV, H_Q, Head_Dim)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, H_Q, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        xq = xq.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        keys = keys.transpose(1, 2)
        # (B, Seq_Len_KV, H_Q, Head_Dim) -> (B, H_Q, Seq_Len_KV, Head_Dim)
        values = values.transpose(1, 2)

        # (B, H_Q, 1, Head_Dim) @ (B, H_Q, Head_Dim, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        # (B, H_Q, 1, Seq_Len_KV) -> (B, H_Q, 1, Seq_Len_KV)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, H_Q, 1, Seq_Len) @ (B, H_Q, Seq_Len_KV, Head_Dim) -> (B, H_Q, 1, Head_Dim)
        output = torch.matmul(scores, values)
        # (B, H_Q, 1, Head_Dim) -> (B, 1, H_Q, Head_Dim) -> (B, 1, Dim)
        output = (output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1))
        return self.o_proj(output) # (B, 1, Dim) -> (B, 1, Dim)



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_layernorm = nn.RMSNorm(config.dim,eps = config.norm_eps)
        self.mlp = FeedForward(config)
        self.post_attention_layernorm = nn.RMSNorm(config.dim,eps = config.norm_eps)
        self.self_attn = CausalAttention(config)
        
    
    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_complex)
        # (B, Seq_Len, Dim) + (B, Seq_Len, Dim) --> (B, Seq_Len, Dim)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out
    
class LlamaTokenEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)

    def forward(self, tokens: torch.tensor):
        #(B, seq_len = T)
        B,T = tokens.shape
        assert T == 1, "Seq len must be 1. Only for inference"
        #(B, T) -> (B,T,dim)
        h = self.embed_tokens(tokens).to(torch.bfloat16)
        return h

class LlamaLayer(nn.Module):
    def __init__(self, config, freqs_complex: torch.Tensor):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([Block(config) for _ in range(1)]) # nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.freqs_complex = freqs_complex
    
    def forward(self, h: torch.tensor, start_pos: int):
        #(B, seq_len = T, Dim)
        B,T,_ = h.shape
        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + T]
        freqs_complex = self.freqs_complex[start_pos:start_pos + T]
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        return h
    
class LlamaFinalLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.RMSNorm(config.dim, eps = config.norm_eps)

    def forward(self, h: torch.tensor):
        h = self.norm(h)
        return h
    
class LlamaFinalLayerHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, h: torch.tensor):
        h = self.lm_head(h)
        return h
    
class LlamaModelFull(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = nn.RMSNorm(config.dim, eps = config.norm_eps)

        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.embed_tokens.weight = self.lm_head.weight
        self.freqs_complex = precompute_theta_pos_frequencies(self.config.dim // self.config.n_heads, self.config.max_position_embeddings * 2, device = self.config.device)
    
    def forward(self, tokens: torch.tensor, start_pos: int):
        #(B, seq_len = T)
        B,T = tokens.shape
        assert T == 1, "Seq len must be 1. Only for inference"
        #(B, T) -> (B,T,dim)
        h = self.embed_tokens(tokens).to(torch.bfloat16)
        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + T]
        freqs_complex = self.freqs_complex[start_pos:start_pos + T]
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.lm_head(h).float()
        return output
    
class LlamaCpuDiskRun():
    def __init__(self,config):
        self.config = config
        self.freqs_complex = precompute_theta_pos_frequencies(self.config.dim // self.config.n_heads, self.config.max_position_embeddings * 2, device = self.config.device)
        self.llamatoken = LlamaTokenEmbed(self.config)
        self.llamalayer = LlamaLayer(self.config,self.freqs_complex)
        self.llamafinalnorm = LlamaFinalLayerNorm(self.config)
        self.llamafinallmhead = LlamaFinalLayerHead(self.config)
        prev_time = time.time()
        self.llamatoken.load_state_dict(load_file(config.model_dir + "/separated_weights/embed_tokens.safetensors"), strict=True)
        print(time.time() - prev_time)
        self.llamafinalnorm.load_state_dict(load_file(config.model_dir + "/separated_weights/norm.safetensors"), strict=True)
        self.llamafinallmhead.load_state_dict(load_file(config.model_dir + "/separated_weights/lm_head.safetensors"), strict=True)

    def run(self,tokens : torch.Tensor, curr_pos: int):
        total_time = time.time()
        x = self.llamatoken(tokens)
        layer_time_avg = 0
        layer_load_t_avg = 0
        for i in range(0,32):
            print(f"layer{i}")
            prev_time = time.time()
            self.llamalayer.load_state_dict(load_file(self.config.model_dir + f"/separated_weights/layers{i}.safetensors"), strict=True)
            t = time.time() - prev_time
            layer_load_t_avg += t
            print(t)
            prev_time = time.time()
            x = self.llamalayer(x,curr_pos)
            t = time.time() - prev_time
            layer_time_avg += t
            print(t)
        print("final layers")
        prev_time = time.time()
        x = self.llamafinallmhead(self.llamafinalnorm(x))
        print(time.time() - prev_time)
        print(x.shape)
        print("total time")
        print(time.time() - total_time)
        print(f"average layer compute and load time:{layer_time_avg/32},{layer_load_t_avg/32}" )
        

def load_parameters_to_layers(model, model_type : str, layer : Optional[int] = None):
    with safe_open("C:/yoghes/llms/Llama-3.1-8B-Instruct/model-00001-of-00004.safetensors", framework = "pt", device = llama_config.device) as f:
        for k in f.keys():
            if("token" in k or "0" in k):
                pass
    print("called")

####--------------------------------------------------------------------------------#########
# Weight tieing done here.
# No.of layers = 1

llama_config = LlamaConfig()
model = LlamaCpuDiskRun(llama_config)
#model = LlamaModelFull(llama_config)
#print("model loaded")
tokenizer = AutoTokenizer.from_pretrained(llama_config.model_dir)
tokens = torch.tensor(tokenizer.encode("")).to(dtype=torch.long)
tokens = tokens.unsqueeze(1)
#print(tokens.shape)     
model.run(tokens,0)