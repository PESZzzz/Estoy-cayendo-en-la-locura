import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
from einops import rearrange
from torch import Tensor, nn

scaled_dot_product_attention = nn.functional.scaled_dot_product_attention
if os.environ.get('USE_SAGEATTN', '0') == '1':
    try:
        from sageattention import sageattn
        scaled_dot_product_attention = sageattn
    except ImportError:
        pass


def attention(q: Tensor, k: Tensor, v: Tensor, **kwargs) -> Tensor:
    x = scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


def timestep_embedding(t: Tensor, dim=256, max_period=10000, time_factor: float = 1000.0):
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class GELU(nn.Module):
    def __init__(self, approximate='tanh'):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: Tensor) -> Tensor:
        return nn.functional.gelu(x, approximate=self.approximate)


class TimeMLPEmbedder(nn.Module):
    def __init__(self, hidden_size: int = 1024, in_dim: int = 256, time_embed_dim: int = 1024):
        super().__init__()
        # in_dim DEBE ser 256 para matchear time_in.in_layer.weight -> [1024, 256] del GGUF
        self.in_layer = nn.Linear(in_dim, time_embed_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(time_embed_dim, hidden_size, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        h = self.silu(self.in_layer(x))
        return self.out_layer(h)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        out_dim: int = 1024,
        num_heads: int = 16,
        qkv_bias: bool = True,
        in_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(in_dim or dim, 3 * dim, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, x: Tensor, pe: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = self.proj(x)
        return x


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int = 1024, double: bool = True, vec_dim: Optional[int] = None):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        actual_vec_dim = vec_dim if vec_dim is not None else dim
        self.lin = nn.Linear(actual_vec_dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(nn.functional.silu(vec))[:, None, :]
        out = out.chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        vec_dim: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        actual_vec_dim = vec_dim if vec_dim is not None else hidden_size
        
        # Stream de Imagen: [6144, 1024]
        self.img_mod = Modulation(hidden_size, double=True, vec_dim=actual_vec_dim)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, out_dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # Stream de Texto / Condición: [6144, 1024]
        self.txt_mod = Modulation(hidden_size, double=True, vec_dim=actual_vec_dim)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, out_dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

    def forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor = None) -> Tuple[Tensor, Tensor]:
        img_mod1, img_mod2 = self.img_mod(vec)
        txt_mod1, txt_mod2 = self.txt_mod(vec)

        img_modulated = self.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = self.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.img_attn.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.txt_attn.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        k = torch.cat((txt_k, img_k), dim=2)
        v = torch.cat((txt_v, img_v), dim=2)

        attn = attention(q, k, v, pe=pe)
        txt_attn, img_attn = attn[:, : txt.shape[1]], attn[:, txt.shape[1]:]

        img = img + img_mod1.gate * self.img_attn.proj(img_attn)
        img_mlp_in = self.img_norm2(img)
        img_mlp_in = (1 + img_mod2.scale) * img_mlp_in + img_mod2.shift
        img = img + img_mod2.gate * self.img_mlp(img_mlp_in)

        txt = txt + txt_mod1.gate * self.txt_attn.proj(txt_attn)
        txt_mlp_in = self.txt_norm2(txt)
        txt_mlp_in = (1 + txt_mod2.scale) * txt_mlp_in + txt_mod2.shift
        txt = txt + txt_mod2.gate * self.txt_mlp(txt_mlp_in)

        return img, txt


class SingleStreamBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        vec_dim: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        actual_vec_dim = vec_dim if vec_dim is not None else hidden_size

        self.linear1 = nn.Linear(hidden_size, 3 * hidden_size + mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False, vec_dim=actual_vec_dim)

    def forward(self, x: Tensor, vec: Tensor, pe: Tensor = None) -> Tensor:
        mod, _ = self.modulation(vec)

        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        lin1_out = self.linear1(x_mod)
        qkv, mlp = lin1_out[..., :3 * self.hidden_dim], lin1_out[..., 3 * self.hidden_dim:]

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn = attention(q, k, v, pe=pe)
        mlp_processed = self.mlp_act(mlp)

        output = self.linear2(torch.cat((attn, mlp_processed), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int = 1024, patch_size: int = 1, out_channels: int = 64):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True) # [64, 1024]
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


class Hunyuan3DDiT(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        context_in_dim: int = 1536,   # EXACTO AL GGUF [1024, 1536]
        hidden_size: int = 1024,      # EXACTO AL GGUF 1024
        mlp_ratio: float = 4.0,
        num_heads: int = 16,
        depth: int = 16,
        depth_single_blocks: int = 32,
        time_factor: float = 1000,
        guidance_embed: bool = False,
        vec_dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.context_in_dim = context_in_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.time_factor = time_factor
        self.out_channels = 64
        self.guidance_embed = guidance_embed

        actual_vec_dim = vec_dim if vec_dim is not None else hidden_size

        # Capas de Entrada
        self.latent_in = nn.Linear(self.in_channels, self.hidden_size, bias=True) # [1024, 64]
        self.time_in = TimeMLPEmbedder(hidden_size=self.hidden_size, in_dim=256, time_embed_dim=1024) # in: [1024, 256], out: [1024, 1024]
        self.cond_in = nn.Linear(self.context_in_dim, self.hidden_size, bias=True) # [1024, 1536]
        
        self.guidance_in = (
            TimeMLPEmbedder(hidden_size=self.hidden_size, in_dim=256, time_embed_dim=1024)
            if guidance_embed else nn.Identity()
        )

        # Bloques de Atencion
        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    vec_dim=actual_vec_dim,
                )
                for _ in range(depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_ratio=mlp_ratio,
                    vec_dim=actual_vec_dim,
                )
                for _ in range(depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels) # [64, 1024]

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        contexts: dict,
        guidance: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        cond = contexts['main'] if isinstance(contexts, dict) and 'main' in contexts else contexts

        latent = self.latent_in(x)
        vec = self.time_in(timestep_embedding(t, dim=256, time_factor=self.time_factor).to(dtype=latent.dtype))

        if self.guidance_embed and guidance is not None:
            vec = vec + self.guidance_in(timestep_embedding(guidance, dim=256, time_factor=self.time_factor).to(dtype=latent.dtype))

        cond = self.cond_in(cond)
        pe = None

        for block in self.double_blocks:
            latent, cond = block(img=latent, txt=cond, vec=vec, pe=pe)

        latent = torch.cat((cond, latent), 1)
        for block in self.single_blocks:
            latent = block(latent, vec=vec, pe=pe)

        latent = latent[:, cond.shape[1]:, ...]
        latent = self.final_layer(latent, vec)
        return latent