## LLaMA - Large Language Model with Attention

import torch
import torch.nn.functional as F
import math
import torch.nn as nn
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32 # Number of heads for the queries
    n_kv_heads: Optional[int] = None # Number of heads for the keys and values. If None, defaults to n_heads
    vocab_size: int = -1 # This will be set when we load the tokenizer
    multiple_of: int = 256 
    ffn_dim_multiplier: Optional[float] = None # If None, defaults to 4.0
    norm_eps: float = 1e-5
    
    # Needed for KV cache
    max_batch_size: int = 32
    max_seq_len: int = 2048

    device: str = None

def precomputed_theta_pos_frequencies(head_dim: int, seq_len: int, device: str, theta: float = 10000.0):
    # As written in the paper, the dimentions o the embedding must be even
    assert head_dim % 2 == 0, "The head_dim must be even"
    # Built the theta parameters
    # According to the formula theta_i = 10000 ^ (-2(i-1)/dim) for i = [1,2,3,..dim/2]
    # Shape: (head_dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2).float()
    # Shape : (head_dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Construct the positions (the "m" parameter)
    # shape: (seq_len)
    m = torch.arange(seq_len, device=device)
    # multiply each theta by each position using the outer product
    # shape : (seq_len) outer_product * (head_dim / 2) -> (seq_len, head_dim / 2)
    freq = torch.outer(m, theta).float()
    # we can computer complex numbers in the polar form c = R * exp(i * m * theta), where R = 1 as follow
    # shape: (seq_len, head_dim/2) -> (seq-len, head_dim/2)
    freq_complex = torch.polar(torch.ones_like(freq), freq)
    return freq_complex

def apply_rotary_embeddings(x: torch.Tensor, freq_complex: torch.Tensor, device: str):
    # We transform the each subsequent pair of tokens into a pair of complex numbers
    # shape : (B, seq_len, head_dim) -> (B, seq_len, h, head_dim / 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # shape : (seq_len, head_dim / 2) -> (1, seq_len, 1, head_dim / 2)
    freq_complex = freq_complex.unsqueeze(0).unsqueeze(2)
    # shape : (B, seq_len, h, head_dim / 2) * (1, seq_len, 1, head_dim / 2) = (B, seq_len, h, head_dim / 2)
    x_rotate = x_complex * freq_complex
    # (B, seq_len, h, head_dim / 2) -> (B, seq_len, h, head_dim/2 ,2)
    x_out = torch.view_as_real(x_rotate)
    # (B, seq_len, h, head_dim/2, 2) -> (B, seq_len, h * head_dim / 2 * 2)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x: torch.Tensor, n_rep: int)-> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        return (
            # (B, seq_len, n_kv_heads, 1, head_dim)
            x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
        )

class SelfAttention(nn.Module): 
    def  __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # Indicates the number of heads for the queries
        self.n_heads_q = args.n_heads
        # Indiates how many times the heads of keys and value should be repeated to match the head of the Query
        self.n_rep = self.n_heads_q // self.n_kv_heads
        # Indicates the dimentiona of each head
        self.head_dim = args.dim // args.n_heads

        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freq_complex: torch.Tensor):
        batch_size, seq_len, _ = x.shape #(B, 1, dim)
        # Apply the wq, wk, wv matrices to query, key and value
        # (B, 1, dim) -> (B, 1, H_q * head_dim)
        xq = self.wq(x)
        # (B, 1, dim) -> (B, 1, H_kv * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        # (B, 1, H_q * head_dim) -> (B, 1, H_q, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        # (B, 1, H_kv * head_dim) -> (B, 1, H_kv, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        # Apply the rotary embeddings to the keys and values
        # Does not chnage the shape of the tensor
        # (B, 1, H_kv, head_dim) -> (B, 1, H_kv, head_dim)
        xq = apply_rotary_embeddings(xq, freq_complex, device=x.device)
        xk = apply_rotary_embeddings(xk, freq_complex, device=x.device)

        # Replace the enty in the cache for this token
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv

        # Retrive all the cached keys and values so far
        # (B, seq_len_kv, H_kv, head_dim)
        keys = self.cache_k[:batch_size, 0:start_pos + seq_len]
        values = self.cache_v[:batch_size, 0:start_pos+seq_len] 

        # Repeat the heads of the K and V to reach the number of heads of the queries
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        # (B, 1, h_q, head_dim) --> (b, h_q, 1, head_dim)
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # (B, h_q, 1, head_dim) @ (B, h_kv, seq_len-kv, head_dim) -> (B, h_q, 1, seq_len-kv)
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # (B, h_q, 1, seq_len) @ (B, h_q, seq_len-kv, head_dim) --> (b, h-q, q, head_dim)
        output = torch.matmul(scores, values)

        # (B, h_q, 1, head_dim) -> (B, 1, h_q, head_dim) -> ()
        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) # (B, 1, dim) -> (B, 1, dim)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # Assuming 'hidden_dim' is calculated as per your specifications
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)  # Applying your specific transformation
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        #hidden_dim = int(2 * hidden_dim / 3)  # Applying your specific transformation
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)  # This layer seems to be missing in your original setup
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)  # Corrected to match checkpoint

    def forward(self, x: torch.Tensor):
        swish = F.silu(self.w1(x))  # Apply first transformation
        x_V = self.w3(x) 
        x = swish * x_V        # Apply contraction to original dimension
        x = self.w2(x)  # Apply optional additional transformation
        return x

class EncoderBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads

        self.attention = SelfAttention(args)
        self.feed_forward = FeedForward(args)

        # normalize BEFORE the self attention
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # Normalization BEFORE the feed forward
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, dim) + (B, seq_len, dim) -> (B, seq_len, dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # The gamma parameter
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        # (B, seq_len, dim) -> (B, seq_len, 1)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # dim : (B, seq_len, dim) -> (B, seq_len, dim)
        return self.weight * self._norm(x.float()).type_as(x)

class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()

        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias=False)

        # To precompute the frequencies of the Rotary Positional Encodings
        self.freqs_complex = precomputed_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device=self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos: int):
        # (B, seq_len)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"
  
        # (B, seq_len) -> (B, seq_len, dim)
        h = self.tok_embeddings(tokens)

        # Retrive the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        # Consecutively apply all the encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h =  self.norm(h)
        output = self.output(h).float()
        return output