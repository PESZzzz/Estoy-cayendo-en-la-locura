# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

import os

import torch
import torch.nn.functional as F

scaled_dot_product_attention = F.scaled_dot_product_attention
if os.environ.get('CA_USE_SAGEATTN', '0') == '1':
    try:
        from sageattention import sageattn
    except ImportError:
        raise ImportError('Please install the package "sageattention" to use this USE_SAGEATTN.')
    scaled_dot_product_attention = sageattn


class CrossAttentionProcessor:
    def __call__(self, attn, q, k, v):
        out = scaled_dot_product_attention(q, k, v)
        return out


class FlashVDMCrossAttentionProcessor:
    def __init__(self, topk=None):
        self.topk = topk

    def __call__(self, attn, q, k, v):
        if k.shape[-2] == 3072:
            topk = 1024
        elif k.shape[-2] == 512:
            topk = 256
        else:
            topk = k.shape[-2] // 3

        if self.topk is True:
            q1 = q[:, :, ::100, :]
            sim = q1 @ k.transpose(-1, -2)
            sim = torch.mean(sim, -2)
            topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
            topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
            v0 = torch.gather(v, dim=-2, index=topk_ind)
            k0 = torch.gather(k, dim=-2, index=topk_ind)
            out = scaled_dot_product_attention(q, k0, v0)
        elif self.topk is False:
            out = scaled_dot_product_attention(q, k, v)
        else:
            idx, counts = self.topk
            start = 0
            
            # [OPTIMIZACIÓN]: Pre-asignamos el tensor final en la memoria de la GPU
            # en lugar de hacer un outs.append() seguido de un torch.cat()
            out = torch.empty((q.shape[0], q.shape[1], q.shape[2], v.shape[-1]), dtype=q.dtype, device=q.device)
            
            for grid_coord, count in zip(idx, counts):
                end = start + count
                q_chunk = q[:, :, start:end, :]
                k0, v0 = self.select_topkv(q_chunk, k, v, topk)
                
                # Insertamos directamente en el espacio pre-asignado
                out[:, :, start:end, :] = scaled_dot_product_attention(q_chunk, k0, v0)
                
                start += count
                
        self.topk = False
        return out

    def select_topkv(self, q_chunk, k, v, topk):
        q1 = q_chunk[:, :, ::50, :]
        sim = q1 @ k.transpose(-1, -2)
        sim = torch.mean(sim, -2)
        topk_ind = torch.topk(sim, dim=-1, k=topk).indices.squeeze(-2).unsqueeze(-1)
        topk_ind = topk_ind.expand(-1, -1, -1, v.shape[-1])
        v0 = torch.gather(v, dim=-2, index=topk_ind)
        k0 = torch.gather(k, dim=-2, index=topk_ind)
        return k0, v0


class FlashVDMTopMCrossAttentionProcessor(FlashVDMCrossAttentionProcessor):
    def select_topkv(self, q_chunk, k, v, topk):
        q1 = q_chunk[:, :, ::30, :]
        sim = q1 @ k.transpose(-1, -2)
        sim = sim.softmax(-1)
        sim = torch.mean(sim, 1)
        activated_token = torch.where(sim > 1e-6)[2]
        index = torch.unique(activated_token, return_counts=True)[0].unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        index = index.expand(-1, v.shape[1], -1, v.shape[-1])
        v0 = torch.gather(v, dim=-2, index=index)
        k0 = torch.gather(k, dim=-2, index=index)
        return k0, v0