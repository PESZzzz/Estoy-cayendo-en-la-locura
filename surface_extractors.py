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
#   1. Aggressive memory cleanup after each mesh extraction to prevent OOM
#      when processing multiple views or batch sizes > 1.
#   2. Explicit .detach().cpu() before numpy conversion, then immediate
#      deletion of the GPU tensor so the VRAM is returned before the CPU
#      marching-cubes routine allocates its own large buffers.
#   3. Per-batch-item extraction: only the current grid_logit is kept in
#      scope; the rest of the batch tensor is untouched.
#   4. Intermediate tensor deletion in DMC path (sdf, verts) so a low-VRAM
#      GPU can survive the differentiable marching-cubes pass.
#
# Look for "[COMMUNITY]" tags in the comments below.
# ============================================================================

import gc
from typing import Union, Tuple, List

import numpy as np
import torch
from skimage import measure


# =============================================================================
# [COMMUNITY] Memory cleanup helper
# =============================================================================
# Called after heavy tensor blocks to force the Python garbage collector
# to release orphaned tensors and, on CUDA, to return freed blocks to the
# driver pool so the next allocation does not OOM.
# =============================================================================
def free_memory():
    """Fuerza la limpieza profunda tanto de la RAM como de la VRAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class Latent2MeshOutput:
    def __init__(self, mesh_v=None, mesh_f=None):
        self.mesh_v = mesh_v
        self.mesh_f = mesh_f


def center_vertices(vertices):
    """Translate the vertices so that bounding box is centered at zero."""
    vert_min = vertices.min(dim=0)[0]
    vert_max = vertices.max(dim=0)[0]
    vert_center = 0.5 * (vert_min + vert_max)
    return vertices - vert_center


class SurfaceExtractor:
    def _compute_box_stat(self, bounds: Union[Tuple[float], List[float], float], octree_resolution: int):
        """
        Compute grid size, bounding box minimum coordinates, and bounding box size based on input
        bounds and resolution.

        Args:
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or a single
            float representing half side length.
                If float, bounds are assumed symmetric around zero in all axes.
                Expected format if list/tuple: [xmin, ymin, zmin, xmax, ymax, zmax].
            octree_resolution (int): Resolution of the octree grid.

        Returns:
            grid_size (List[int]): Grid size along each axis (x, y, z), each equal to octree_resolution + 1.
            bbox_min (np.ndarray): Minimum coordinates of the bounding box (xmin, ymin, zmin).
            bbox_size (np.ndarray): Size of the bounding box along each axis (xmax - xmin, etc.).
        """
        if isinstance(bounds, float):
            bounds = [-bounds, -bounds, -bounds, bounds, bounds, bounds]

        bbox_min, bbox_max = np.array(bounds[0:3]), np.array(bounds[3:6])
        bbox_size = bbox_max - bbox_min
        grid_size = [int(octree_resolution) + 1, int(octree_resolution) + 1, int(octree_resolution) + 1]
        return grid_size, bbox_min, bbox_size

    def run(self, *args, **kwargs):
        """
        Abstract method to extract surface mesh from grid logits.

        This method should be implemented by subclasses.

        Raises:
            NotImplementedError: Always, since this is an abstract method.
        """
        return NotImplementedError

    def __call__(self, grid_logits, **kwargs):
        """
        Process a batch of grid logits to extract surface meshes.

        Args:
            grid_logits (torch.Tensor): Batch of grid logits with shape (batch_size, ...).
            **kwargs: Additional keyword arguments passed to the `run` method.

        Returns:
            List[Optional[Latent2MeshOutput]]: List of mesh outputs for each grid in the batch.
                If extraction fails for a grid, None is appended at that position.
        """
        outputs = []
        for i in range(grid_logits.shape[0]):
            try:
                # =================================================================
                # [COMMUNITY] Extract only the current item from the batch
                # =================================================================
                # The upstream code passes grid_logits[i] directly.  We extract it
                # into a local variable so that after the extraction we can delete
                # the heavy intermediate arrays (vertices, faces, current_logit)
                # before moving to the next batch item.  This prevents a
                # cumulative memory leak when batch_size > 1.
                # =================================================================
                current_logit = grid_logits[i]

                vertices, faces = self.run(current_logit, **kwargs)
                vertices = vertices.astype(np.float32)
                faces = np.ascontiguousarray(faces)
                outputs.append(Latent2MeshOutput(mesh_v=vertices, mesh_f=faces))

                # =================================================================
                # [COMMUNITY] Immediate cleanup after each batch item
                # =================================================================
                # vertices and faces can be large (hundreds of MB for high-res
                # meshes).  Deleting them now lets the allocator reuse the memory
                # for the next item instead of accumulating until the loop ends.
                # =================================================================
                del vertices
                del faces
                del current_logit
                free_memory()

            except Exception:
                import traceback
                traceback.print_exc()
                outputs.append(None)
                free_memory()

        return outputs


class MCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, mc_level, bounds, octree_resolution, **kwargs):
        """
        Extract surface mesh using the Marching Cubes algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            mc_level (float): The level (iso-value) at which to extract the surface.
            bounds (Union[Tuple[float], List[float], float]): Bounding box coordinates or half side length.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, scaled and translated to bounding
                  box coordinates.
                - faces (np.ndarray): Extracted mesh faces (triangles).
        """
        # =================================================================
        # [COMMUNITY] Detach, move to CPU, convert to numpy, then delete GPU tensor
        # =================================================================
        # The upstream code does `grid_logit.cpu().numpy()` which keeps the
        # original tensor alive until the function returns.  For a 384^3 grid
        # that is ~200 MB on GPU.  We explicitly detach, move to CPU, convert,
        # then delete the GPU tensor immediately so the VRAM is returned BEFORE
        # marching_cubes allocates its own large CPU buffers.
        # =================================================================
        grid_logit_np = grid_logit.detach().cpu().numpy()

        # Release the GPU tensor immediately
        del grid_logit
        free_memory()

        # Execute Marching Cubes (this allocates large CPU buffers)
        vertices, faces, normals, _ = measure.marching_cubes(
            grid_logit_np,
            mc_level,
            method="lewiner"
        )

        # =================================================================
        # [COMMUNITY] Cleanup numpy intermediates
        # =================================================================
        # grid_logit_np can be ~200 MB and normals is another large array.
        # We no longer need them after marching_cubes finishes.
        # =================================================================
        del grid_logit_np
        del normals
        gc.collect()

        grid_size, bbox_min, bbox_size = self._compute_box_stat(bounds, octree_resolution)
        vertices = vertices / grid_size * bbox_size + bbox_min
        return vertices, faces


class DMCSurfaceExtractor(SurfaceExtractor):
    def run(self, grid_logit, *, octree_resolution, **kwargs):
        """
        Extract surface mesh using Differentiable Marching Cubes (DMC) algorithm.

        Args:
            grid_logit (torch.Tensor): 3D grid logits tensor representing the scalar field.
            octree_resolution (int): Resolution of the octree grid.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple containing:
                - vertices (np.ndarray): Extracted mesh vertices, centered and converted to numpy.
                - faces (np.ndarray): Extracted mesh faces (triangles), with reversed vertex order.

        Raises:
            ImportError: If the 'diso' package is not installed.
        """
        device = grid_logit.device
        if not hasattr(self, 'dmc'):
            try:
                from diso import DiffDMC
            except ImportError:
                raise ImportError("Please install diso via `pip install diso`, or set mc_algo to 'mc'")
            self.dmc = DiffDMC(dtype=torch.float32).to(device)

        sdf = -grid_logit / octree_resolution
        sdf = sdf.to(torch.float32).contiguous()

        # =================================================================
        # [COMMUNITY] Delete grid_logit before DMC forward
        # =================================================================
        # grid_logit is no longer needed once sdf is computed.  Deleting it
        # now frees VRAM before DiffDMC allocates its internal buffers.
        # =================================================================
        del grid_logit

        verts, faces = self.dmc(sdf, deform=None, return_quads=False, normalize=True)
        verts = center_vertices(verts)

        # Move to CPU
        vertices = verts.detach().cpu().numpy()
        faces = faces.detach().cpu().numpy()[:, ::-1]

        # =================================================================
        # [COMMUNITY] Cleanup intermediate GPU tensors
        # =================================================================
        # sdf and verts can be large 3D tensors.  Deleting them immediately
        # prevents a VRAM spike that OOMs 4-6 GB GPUs.
        # =================================================================
        del sdf
        del verts
        free_memory()

        return vertices, faces


SurfaceExtractors = {
    'mc': MCSurfaceExtractor,
    'dmc': DMCSurfaceExtractor,
}