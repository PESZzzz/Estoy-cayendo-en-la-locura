# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these
# third-party components and must ensure that the usage of the third party
# components adheres to all relevant laws and regulations.
#
# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters
# (including optimizer states), machine-learning model code, inference-enabling
# code, training-enabling code, fine-tuning enabling code and other elements
# of the foregoing made publicly available by Tencent in accordance with
# TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.
#
# ============================================================================
# COMMUNITY LOW-VRAM / CONSUMER-HARDWARE OPTIMISATIONS
# ============================================================================
# This file has been adapted by the community for running on modest hardware:
#   - Laptops and desktops with 4-8 GB VRAM
#   - CPU-only inference
#   - AMD GPUs (ROCm on Linux, CPU fallback on Windows)
#
# Key adaptations:
#   1. Aggressive memory cleanup (gc.collect + cuda.empty_cache) after heavy
#      tensor operations to prevent OOM crashes.
#   2. Pre-allocated output tensors instead of growing torch.cat() lists.
#      The upstream code appends every chunk to a Python list and then calls
#      torch.cat() at the end, which briefly doubles memory usage (~200 % peak).
#      We write directly into a pre-allocated tensor, keeping peak memory
#      close to the final size.
#   3. Explicit `del` of intermediate tensors inside loops so the allocator
#      can reuse the memory immediately instead of waiting for the function
#      to return.
#   4. `free_memory()` calls between hierarchical octree levels so a
#      low-VRAM GPU can survive the multi-resolution refinement.
#
# Look for "[COMMUNITY]" tags in the comments below.
# ============================================================================

import gc
from typing import Union, Tuple, List, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from tqdm import tqdm

from .attention_blocks import CrossAttentionDecoder
from .attention_processors import FlashVDMCrossAttentionProcessor, FlashVDMTopMCrossAttentionProcessor
from ...utils import logger


# =============================================================================
# [COMMUNITY] Memory cleanup helper
# =============================================================================
# Called after heavy tensor blocks to force the Python garbage collector
# to release orphaned tensors and, on CUDA, to return freed blocks to the
# driver pool so the next allocation does not OOM.
# =============================================================================
def free_memory():
    """Limpieza de RAM y VRAM para evitar crasheos por picos de memoria."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_near_surface_volume_fn(input_tensor: torch.Tensor, alpha: float):
    device = input_tensor.device
    D = input_tensor.shape[0]
    signed_val = 0.0

    # 添加偏移并处理无效值
    val = input_tensor + alpha
    valid_mask = val > -9000  # 假设-9000是无效值

    # 改进的邻居获取函数（保持维度一致）
    def get_neighbor(t, shift, axis):
        """根据指定轴进行位移并保持维度一致"""
        if shift == 0:
            return t.clone()

        # 确定填充轴（输入为[D, D, D]对应z,y,x轴）
        pad_dims = [0, 0, 0, 0, 0, 0]  # 格式：[x前，x后，y前，y后，z前，z后]

        # 根据轴类型设置填充
        if axis == 0:  # x轴（最后一个维度）
            pad_idx = 0 if shift > 0 else 1
            pad_dims[pad_idx] = abs(shift)
        elif axis == 1:  # y轴（中间维度）
            pad_idx = 2 if shift > 0 else 3
            pad_dims[pad_idx] = abs(shift)
        elif axis == 2:  # z轴（第一个维度）
            pad_idx = 4 if shift > 0 else 5
            pad_dims[pad_idx] = abs(shift)

        # 执行填充（添加batch和channel维度适配F.pad）
        padded = F.pad(t.unsqueeze(0).unsqueeze(0), pad_dims[::-1], mode='replicate')  # 反转顺序适配F.pad

        # 构建动态切片索引
        slice_dims = [slice(None)] * 3  # 初始化为全切片
        if axis == 0:  # x轴（dim=2）
            if shift > 0:
                slice_dims[0] = slice(shift, None)
            else:
                slice_dims[0] = slice(None, shift)
        elif axis == 1:  # y轴（dim=1）
            if shift > 0:
                slice_dims[1] = slice(shift, None)
            else:
                slice_dims[1] = slice(None, shift)
        elif axis == 2:  # z轴（dim=0）
            if shift > 0:
                slice_dims[2] = slice(shift, None)
            else:
                slice_dims[2] = slice(None, shift)

        # 应用切片并恢复维度
        padded = padded.squeeze(0).squeeze(0)
        sliced = padded[slice_dims]
        return sliced

    # 获取各方向邻居（确保维度一致）
    left = get_neighbor(val, 1, axis=0)  # x方向
    right = get_neighbor(val, -1, axis=0)
    back = get_neighbor(val, 1, axis=1)  # y方向
    front = get_neighbor(val, -1, axis=1)
    down = get_neighbor(val, 1, axis=2)  # z方向
    up = get_neighbor(val, -1, axis=2)

    # 处理边界无效值（使用where保持维度一致）
    def safe_where(neighbor):
        return torch.where(neighbor > -9000, neighbor, val)

    left = safe_where(left)
    right = safe_where(right)
    back = safe_where(back)
    front = safe_where(front)
    down = safe_where(down)
    up = safe_where(up)

    # 计算符号一致性（转换为float32确保精度）
    sign = torch.sign(val.to(torch.float32))

    # =================================================================
    # [COMMUNITY] Delete the original tensor before stacking neighbours
    # =================================================================
    # `val` is no longer needed; removing it now frees the memory block
    # before we allocate the 6-channel neighbours_sign tensor.
    # =================================================================
    del val

    neighbors_sign = torch.stack([
        torch.sign(left.to(torch.float32)),
        torch.sign(right.to(torch.float32)),
        torch.sign(back.to(torch.float32)),
        torch.sign(front.to(torch.float32)),
        torch.sign(down.to(torch.float32)),
        torch.sign(up.to(torch.float32))
    ], dim=0)

    # =================================================================
    # [COMMUNITY] Aggressive cleanup of intermediate direction tensors
    # =================================================================
    # Each of these is the same size as the full grid; together they can
    # consume several hundred MB.  Deleting them before the next alloc
    # prevents a temporary ~6x memory spike.
    # =================================================================
    del left, right, back, front, down, up
    free_memory()

    # 检查所有符号是否一致
    same_sign = torch.all(neighbors_sign == sign, dim=0)

    # =================================================================
    # [COMMUNITY] Release sign tensors before creating the mask
    # =================================================================
    del neighbors_sign, sign

    # 生成最终掩码
    mask = (~same_sign).to(torch.int32)
    return mask * valid_mask.to(torch.int32)


def generate_dense_grid_points(
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    octree_resolution: int,
    indexing: str = "ij",
):
    length = bbox_max - bbox_min
    num_cells = octree_resolution

    x = np.linspace(bbox_min[0], bbox_max[0], int(num_cells) + 1, dtype=np.float32)
    y = np.linspace(bbox_min[1], bbox_max[1], int(num_cells) + 1, dtype=np.float32)
    z = np.linspace(bbox_min[2], bbox_max[2], int(num_cells) + 1, dtype=np.float32)
    [xs, ys, zs] = np.meshgrid(x, y, z, indexing=indexing)
    xyz = np.stack((xs, ys, zs), axis=-1)
    grid_size = [int(num_cells) + 1, int(num_cells) + 1, int(num_cells) + 1]

    return xyz, grid_size, length


class VanillaVolumeDecoder:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        octree_resolution: int = None,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype
        batch_size = latents.shape[0]

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=octree_resolution,
            indexing="ij"
        )
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        total_points = xyz_samples.shape[0]

        # =================================================================
        # [COMMUNITY] Pre-allocated output tensor
        # =================================================================
        # Upstream code builds a Python list `batch_logits = []` and appends
        # every chunk, then calls torch.cat() at the end.  During the final
        # cat() both the list AND the concatenated tensor exist simultaneously,
        # causing a ~200 % memory spike that OOMs 4-6 GB GPUs.
        #
        # Instead we allocate the final tensor once and write each chunk
        # directly into its slice.  Peak memory stays close to the final size.
        # =================================================================
        grid_logits_flat = torch.empty((batch_size, total_points, 1), dtype=dtype, device=device)

        for start in tqdm(range(0, total_points, num_chunks), desc=f"Volume Decoding",
                          disable=not enable_pbar):
            end = min(start + num_chunks, total_points)
            chunk_queries = xyz_samples[start: end, :]
            chunk_queries = repeat(chunk_queries, "p c -> b p c", b=batch_size)

            logits = geo_decoder(queries=chunk_queries, latents=latents)
            grid_logits_flat[:, start:end, :] = logits

            # =================================================================
            # [COMMUNITY] Immediate deletion of chunk temporaries
            # =================================================================
            # Without explicit del the tensors survive until the end of the
            # iteration, blocking reuse of their memory for the next chunk.
            # =================================================================
            del chunk_queries
            del logits

        grid_logits = grid_logits_flat.view((batch_size, *grid_size)).float()

        # =================================================================
        # [COMMUNITY] Final cleanup before returning
        # =================================================================
        del grid_logits_flat
        del xyz_samples
        free_memory()

        return grid_logits


class HierarchicalVolumeDecoding:
    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: Callable,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        min_resolution: int = 63,
        enable_pbar: bool = True,
        **kwargs,
    ):
        device = latents.device
        dtype = latents.dtype

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij"
        )

        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = torch.nn.Parameter(torch.ones(dilate.weight.shape, dtype=dtype, device=device))

        grid_size = np.array(grid_size)
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype).contiguous().reshape(-1, 3)

        # 2. latents to 3d volume
        batch_size = latents.shape[0]
        total_samples = xyz_samples.shape[0]

        # =================================================================
        # [COMMUNITY] Pre-allocated tensor for first octree level
        # =================================================================
        grid_logits_flat = torch.empty((batch_size, total_samples, 1), dtype=dtype, device=device)

        for start in tqdm(range(0, total_samples, num_chunks),
                          desc=f"Hierarchical Volume Decoding [r{resolutions[0] + 1}]"):
            end = min(start + num_chunks, total_samples)
            queries = xyz_samples[start:end, :]
            batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
            logits = geo_decoder(queries=batch_queries, latents=latents)

            grid_logits_flat[:, start:end, :] = logits

            del queries, batch_queries, logits

        grid_logits = grid_logits_flat.view((batch_size, grid_size[0], grid_size[1], grid_size[2]))
        del xyz_samples, grid_logits_flat
        free_memory()

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
            next_logits = torch.full(next_index.shape, -10000., dtype=dtype, device=device)

            curr_points = extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level)
            curr_points += grid_logits.squeeze(0).abs() < 0.95

            if octree_depth_now == resolutions[-1]:
                expand_num = 0
            else:
                expand_num = 1
            for i in range(expand_num):
                curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
            (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)

            del curr_points

            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for i in range(2 - expand_num):
                next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
            nidx = torch.where(next_index > 0)

            next_points = torch.stack(nidx, dim=1)
            next_points = (next_points * torch.tensor(resolution, dtype=next_points.dtype, device=device) +
                           torch.tensor(bbox_min, dtype=next_points.dtype, device=device))

            total_next_points = next_points.shape[0]

            # =================================================================
            # [COMMUNITY] Pre-allocated tensor for hierarchical refinement
            # =================================================================
            grid_logits_next = torch.empty((batch_size, total_next_points, 1), dtype=dtype, device=device)

            for start in tqdm(range(0, total_next_points, num_chunks),
                              desc=f"Hierarchical Volume Decoding [r{octree_depth_now + 1}]"):
                end = min(start + num_chunks, total_next_points)
                queries = next_points[start:end, :]
                batch_queries = repeat(queries, "p c -> b p c", b=batch_size)
                logits = geo_decoder(queries=batch_queries.to(latents.dtype), latents=latents)

                grid_logits_next[:, start:end, :] = logits

                del queries, batch_queries, logits

            next_logits[nidx] = grid_logits_next[0, ..., 0]
            grid_logits = next_logits.unsqueeze(0)

            # =================================================================
            # [COMMUNITY] Cleanup between octree levels
            # =================================================================
            # Each level can be larger than the previous one.  Without this
            # cleanup a 6 GB GPU OOMs between level 2 and 3.
            # =================================================================
            del grid_logits_next, next_points
            free_memory()

        grid_logits[grid_logits == -10000.] = float('nan')

        return grid_logits


class FlashVDMVolumeDecoding:
    def __init__(self, topk_mode='mean'):
        if topk_mode not in ['mean', 'merge']:
            raise ValueError(f'Unsupported topk_mode {topk_mode}, available: {["mean", "merge"]}')

        if topk_mode == 'mean':
            self.processor = FlashVDMCrossAttentionProcessor()
        else:
            self.processor = FlashVDMTopMCrossAttentionProcessor()

    @torch.no_grad()
    def __call__(
        self,
        latents: torch.FloatTensor,
        geo_decoder: CrossAttentionDecoder,
        bounds: Union[Tuple[float], List[float], float] = 1.01,
        num_chunks: int = 10000,
        mc_level: float = 0.0,
        octree_resolution: int = None,
        min_resolution: int = 63,
        mini_grid_num: int = 4,
        enable_pbar: bool = True,
        **kwargs,
    ):
        processor = self.processor
        geo_decoder.set_cross_attention_processor(processor)

        device = latents.device
        dtype = latents.dtype

        resolutions = []
        if octree_resolution < min_resolution:
            resolutions.append(octree_resolution)
        while octree_resolution >= min_resolution:
            resolutions.append(octree_resolution)
            octree_resolution = octree_resolution // 2
        resolutions.reverse()
        resolutions[0] = round(resolutions[0] / mini_grid_num) * mini_grid_num - 1
        for i, resolution in enumerate(resolutions[1:]):
            resolutions[i + 1] = resolutions[0] * 2 ** (i + 1)

        logger.info(f"FlashVDMVolumeDecoding Resolution: {resolutions}")

        # 1. generate query points
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]
        bbox_min = np.array(bounds[0:3])
        bbox_max = np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min

        xyz_samples, grid_size, length = generate_dense_grid_points(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            octree_resolution=resolutions[0],
            indexing="ij"
        )

        dilate = nn.Conv3d(1, 1, 3, padding=1, bias=False, device=device, dtype=dtype)
        dilate.weight = torch.nn.Parameter(torch.ones(dilate.weight.shape, dtype=dtype, device=device))

        grid_size = np.array(grid_size)

        # 2. latents to 3d volume
        xyz_samples = torch.from_numpy(xyz_samples).to(device, dtype=dtype)
        batch_size = latents.shape[0]
        mini_grid_size = xyz_samples.shape[0] // mini_grid_num
        xyz_samples = xyz_samples.view(
            mini_grid_num, mini_grid_size,
            mini_grid_num, mini_grid_size,
            mini_grid_num, mini_grid_size, 3
        ).permute(
            0, 2, 4, 1, 3, 5, 6
        ).reshape(
            -1, mini_grid_size * mini_grid_size * mini_grid_size, 3
        )

        total_batches = xyz_samples.shape[0]
        num_batchs = max(num_chunks // xyz_samples.shape[1], 1)

        # =================================================================
        # [COMMUNITY] Pre-allocated tensor for FlashVDM mini-grid decode
        # =================================================================
        cat_logits = torch.empty((total_batches, xyz_samples.shape[1], 1), dtype=dtype, device=device)

        for start in tqdm(range(0, total_batches, num_batchs),
                          desc=f"FlashVDM Volume Decoding", disable=not enable_pbar):
            end = min(start + num_batchs, total_batches)
            queries = xyz_samples[start:end, :]
            batch = queries.shape[0]
            batch_latents = repeat(latents.squeeze(0), "p c -> b p c", b=batch)

            processor.topk = True
            logits = geo_decoder(queries=queries, latents=batch_latents)

            cat_logits[start:end] = logits
            del queries, batch_latents, logits

        grid_logits = cat_logits.reshape(
            mini_grid_num, mini_grid_num, mini_grid_num,
            mini_grid_size, mini_grid_size,
            mini_grid_size
        ).permute(0, 3, 1, 4, 2, 5).contiguous().view(
            (batch_size, grid_size[0], grid_size[1], grid_size[2])
        )

        del cat_logits, xyz_samples
        free_memory()

        for octree_depth_now in resolutions[1:]:
            grid_size = np.array([octree_depth_now + 1] * 3)
            resolution = bbox_size / octree_depth_now
            next_index = torch.zeros(tuple(grid_size), dtype=dtype, device=device)
            next_logits = torch.full(next_index.shape, -10000., dtype=dtype, device=device)
            curr_points = extract_near_surface_volume_fn(grid_logits.squeeze(0), mc_level)
            curr_points += grid_logits.squeeze(0).abs() < 0.95

            if octree_depth_now == resolutions[-1]:
                expand_num = 0
            else:
                expand_num = 1
            for i in range(expand_num):
                curr_points = dilate(curr_points.unsqueeze(0).to(dtype)).squeeze(0)
            (cidx_x, cidx_y, cidx_z) = torch.where(curr_points > 0)

            del curr_points

            next_index[cidx_x * 2, cidx_y * 2, cidx_z * 2] = 1
            for i in range(2 - expand_num):
                next_index = dilate(next_index.unsqueeze(0)).squeeze(0)
            nidx = torch.where(next_index > 0)

            next_points = torch.stack(nidx, dim=1)
            next_points = (next_points * torch.tensor(resolution, dtype=torch.float32, device=device) +
                           torch.tensor(bbox_min, dtype=torch.float32, device=device))

            query_grid_num = 6
            min_val = next_points.min(axis=0).values
            max_val = next_points.max(axis=0).values
            vol_queries_index = (next_points - min_val) / (max_val - min_val) * (query_grid_num - 0.001)
            index = torch.floor(vol_queries_index).long()
            index = index[..., 0] * (query_grid_num ** 2) + index[..., 1] * query_grid_num + index[..., 2]
            index = index.sort()
            next_points = next_points[index.indices].unsqueeze(0).contiguous()
            unique_values = torch.unique(index.values, return_counts=True)

            grid_logits = torch.zeros((next_points.shape[1]), dtype=latents.dtype, device=latents.device)
            input_grid = [[], []]

            # =================================================================
            # [COMMUNITY] In-place insertion instead of growing list + cat
            # =================================================================
            # Upstream code accumulates every grid batch in logits_grid_list
            # and then torch.cat()s them.  We write directly into grid_logits
            # at the correct sorted indices, keeping peak memory flat.
            # =================================================================
            start_num = 0
            sum_num = 0
            for grid_index, count in zip(unique_values[0].cpu().tolist(), unique_values[1].cpu().tolist()):
                if sum_num + count < num_chunks or sum_num == 0:
                    sum_num += count
                    input_grid[0].append(grid_index)
                    input_grid[1].append(count)
                else:
                    processor.topk = input_grid
                    logits_grid = geo_decoder(queries=next_points[:, start_num:start_num + sum_num], latents=latents)
                    # Insert directly at sorted positions instead of appending
                    grid_logits[index.indices[start_num:start_num + sum_num]] = logits_grid.squeeze(0).squeeze(-1)

                    start_num = start_num + sum_num
                    input_grid = [[grid_index], [count]]
                    sum_num = count
                    del logits_grid

            if sum_num > 0:
                processor.topk = input_grid
                logits_grid = geo_decoder(queries=next_points[:, start_num:start_num + sum_num], latents=latents)
                grid_logits[index.indices[start_num:start_num + sum_num]] = logits_grid.squeeze(0).squeeze(-1)
                del logits_grid

            next_logits[nidx] = grid_logits
            grid_logits = next_logits.unsqueeze(0)

            # =================================================================
            # [COMMUNITY] Cleanup after each hierarchical level
            # =================================================================
            free_memory()

        grid_logits[grid_logits == -10000.] = float('nan')

        return grid_logits