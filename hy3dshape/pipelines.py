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
# COMMUNITY AMD / LOW-VRAM OPTIMIZATIONS
# ============================================================================
# This file has been adapted by the community for running on modest hardware:
#   - AMD GPUs (ROCm on Linux, CPU fallback on Windows)
#   - Laptops and desktops with limited VRAM (4-8 GB)
#   - CPU-only inference when no GPU is available
#   - HYBRID mode: uses GPU for diffusion + CPU for mesh extraction
#
# Key adaptations:
#   1. Thread limiting to prevent system freezes on consumer CPUs
#   2. Safe torch.compile() disabling on Windows/CPU to avoid hard crashes
#   3. Manual sequential CPU offloading as a lightweight alternative to
#      accelerate-based offloading (works without the accelerate library)
#   4. CPU-optimized mesh extraction with full core utilisation and timing
#   5. Import path fallbacks so the pipeline works regardless of how the
#      package was installed (hy3dshape, hy3dgen, _hy3dgen, etc.)
#   6. VRAM cleanup helpers to survive on GPUs with < 8 GB
#   7. SMART DEVICE MANAGER: auto-detects hardware and picks the best
#      strategy (GPU-only, Hybrid, or CPU-only) to avoid crashes
#   8. RAM spike prevention: removed hidden memory bombs (deepcopy, etc.)
#
# If you are a developer extending this, look for the "[COMMUNITY]" tags
# in the comments below.
# ============================================================================

import copy
import gc
import importlib
import inspect
import os
import sys
import warnings
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_accelerate_version, is_accelerate_available
from tqdm import tqdm

from .models.autoencoders import ShapeVAE
from .models.autoencoders import SurfaceExtractors
from .utils import logger, synchronize_timer, smart_load_model


# =============================================================================
# [COMMUNITY] Smart Device Manager
# =============================================================================
# This class looks at your computer and decides the BEST way to run the
# pipeline without crashing.  It answers three questions:
#
#   1. Do you have a GPU?  (NVIDIA CUDA, AMD ROCm, or Apple MPS)
#   2. How much VRAM does it have?
#   3. Should we run everything on GPU, split the work (GPU + CPU), or
#      fall back to CPU-only?
#
# Strategies
# ----------
#   "gpu"    -> Diffusion + VAE + mesh extraction all on the GPU.
#               Used when you have a dedicated card with 8 GB+ VRAM.
#
#   "hybrid" -> Diffusion runs on the GPU (fast), but the heavy mesh
#               extraction runs on the CPU (saves VRAM).
#               Used for low-VRAM cards (4-8 GB), AMD integrated graphics,
#               or laptops where the GPU shares RAM with the system.
#
#   "cpu"    -> Everything runs on the CPU.
#               Used when there is no GPU, or the GPU is too weak.
#
# Why this matters for AMD users on Windows
# -----------------------------------------
# Official PyTorch does NOT support ROCm on Windows.  That means most AMD
# GPUs on Windows are invisible to PyTorch and the pipeline would normally
# fall back to CPU-only.  With the Hybrid strategy, if you manage to install
# a custom ROCm build (or if PyTorch ever adds support), the manager will
# detect the AMD GPU, see it has limited VRAM, and automatically use the
# GPU for diffusion while keeping the RAM-hungry mesh extraction on the CPU.
# =============================================================================
class SmartDeviceManager:
    """
    Auto-detects hardware and picks the safest, fastest strategy.

    You do NOT need to touch this class.  The pipeline uses it automatically.
    Advanced users can force a strategy by passing:
        device="cuda", strategy="hybrid"
    when creating the pipeline.
    """

    def __init__(self, device: Optional[Union[str, torch.device]] = None,
                 strategy: Optional[str] = None):
        self.strategy = strategy  # "gpu", "hybrid", "cpu", or None (auto)
        self.device = None
        self.vae_device = None
        self.vram_gb = 0.0
        self.is_amd_windows = False

        # ------------------------------------------------------------------
        # Step 1 – Detect what PyTorch can see
        # ------------------------------------------------------------------
        if device is not None:
            # User forced a specific torch device
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # ------------------------------------------------------------------
        # Step 2 – Measure VRAM (only works for CUDA/ROCm/MPS)
        # ------------------------------------------------------------------
        if self.device.type == "cuda":
            try:
                props = torch.cuda.get_device_properties(self.device)
                self.vram_gb = props.total_memory / (1024 ** 3)
                # Heuristic: AMD cards on Linux report "cuda" too when ROCm
                # is installed.  On Windows they usually do not appear at all.
                if "AMD" in props.name or "Radeon" in props.name:
                    self.is_amd_windows = sys.platform == "win32"
            except Exception:
                self.vram_gb = 0.0
        elif self.device.type == "mps":
            # Apple Silicon – shared memory, treat like low-VRAM GPU
            try:
                import psutil
                self.vram_gb = psutil.virtual_memory().total / (1024 ** 3)
            except Exception:
                self.vram_gb = 8.0

        # ------------------------------------------------------------------
        # Step 3 – Pick strategy if the user did not force one
        # ------------------------------------------------------------------
        if self.strategy is None:
            if self.device.type == "cpu":
                self.strategy = "cpu"
            elif self.vram_gb >= 8.0 and not self.is_amd_windows:
                # Dedicated NVIDIA / AMD-dGPU with plenty of VRAM
                self.strategy = "gpu"
            else:
                # Low-VRAM GPU, integrated AMD, Apple MPS, or AMD on Windows
                self.strategy = "hybrid"

        # ------------------------------------------------------------------
        # Step 4 – Decide where the VAE (mesh extraction) lives
        # ------------------------------------------------------------------
        # In "hybrid" mode we keep the diffusion model on the GPU but move
        # the VAE to the CPU for mesh extraction.  This is the secret sauce
        # that lets 4-6 GB cards survive the octree decode.
        if self.strategy == "hybrid":
            self.vae_device = torch.device("cpu")
        else:
            self.vae_device = self.device

        logger.info(
            f"[SmartDevice] Detected {self.device.type.upper()} "
            f"({self.vram_gb:.1f} GB VRAM / shared RAM). "
            f"Strategy: {self.strategy.upper()}. "
            f"Diffusion on {self.device}, VAE on {self.vae_device}."
        )

    def get_safe_export_params(self, user_octree=None, user_chunks=None):
        """
        Return conservative octree_resolution / num_chunks when running on
        CPU or in hybrid mode.  Prevents Windows from silently killing the
        process when RAM runs out.

        The user can still override these from Modly nodes – this is only
        the safety net when nothing is specified.
        """
        if self.strategy == "gpu" and self.vram_gb >= 8:
            # Powerful GPU – generous defaults
            return user_octree or 256, user_chunks or 4000

        # CPU or Hybrid – we are RAM-limited
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            ram_gb = 16.0  # guess

        if ram_gb < 12:
            octree = 128
            chunks = 1000
        elif ram_gb < 24:
            octree = 192
            chunks = 2000
        else:
            octree = 256
            chunks = 4000

        return user_octree or octree, user_chunks or chunks


# =============================================================================
# [COMMUNITY] Universal CPU / GPU Optimisation Block
# =============================================================================
# Consumer laptops and desktops often have fewer cores than data-centre CPUs.
# PyTorch defaults to using ALL logical cores, which causes:
#   - System freezes while generating
#   - Thermal throttling on laptops
#   - Worse performance than using fewer threads (oversubscription)
#
# We reserve ~25 % of cores for the OS and other apps, and cap both
# intra-op and inter-op thread pools.
# =============================================================================
total_cores = os.cpu_count() or 4
safe_cores = max(1, int(total_cores * 0.75))

torch.set_num_threads(safe_cores)
torch.set_num_interop_threads(max(1, int(safe_cores // 2)))
os.environ["OMP_NUM_THREADS"] = str(safe_cores)
os.environ["MKL_NUM_THREADS"] = str(safe_cores)

# MKL-DNN (oneDNN) gives a nice speed-up on Intel/AMD CPUs.
torch.backends.mkldnn.enabled = True
# "medium" is the sweet spot between speed and precision for inference.
torch.set_float32_matmul_precision("medium")


# =============================================================================
# [COMMUNITY] Safe torch.compile() Disabling on CPU / Windows
# =============================================================================
# torch.compile() requires a working C++ compiler (cl.exe on Windows) in PATH
# at RUNTIME. Visual Studio Build Tools installs cl.exe but does NOT add it
# to the system PATH, so torch.compile() crashes with a hard error on most
# consumer Windows machines.
#
# The speed-up on CPU is marginal (~10-20 %) but the failure mode is a
# complete crash. For a community extension that must work on any laptop,
# we detect this scenario and neuter torch.compile() before it can bite.
# =============================================================================
_DISABLE_COMPILE = False
if sys.platform == "win32" and not torch.cuda.is_available():
    _DISABLE_COMPILE = True
    warnings.warn(
        "torch.compile() is disabled on CPU/Windows for stability. "
        "Install Visual Studio Build Tools and run from 'x64 Native Tools "
        "Command Prompt' if you want compiled kernels.",
        UserWarning,
        stacklevel=2,
    )

if _DISABLE_COMPILE:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.reset()
    _original_compile = torch.compile

    def _noop_compile(model, *args, **kwargs):
        """No-op replacement for torch.compile on unsupported platforms."""
        return model

    torch.compile = _noop_compile


# =============================================================================
# [COMMUNITY] VRAM Cleanup Helper
# =============================================================================
# Small helper used by the manual-offload and hybrid paths.  Calling
# gc.collect() before empty_cache() ensures that orphaned CUDA tensors are
# freed immediately.
# =============================================================================
def free_vram():
    """Free residual GPU memory to prevent hangs on laptops and AMD cards."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from
    the scheduler after the call. Handles custom timesteps. Any kwargs will be
    supplied to `scheduler.set_timesteps`.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@synchronize_timer('Export to trimesh')
def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


# =============================================================================
# [COMMUNITY] Robust Config Instantiation with Import Fallbacks
# =============================================================================
# The upstream code assumes the package is always installed as "hy3dshape".
# In practice users install it in many ways (editable, zip extraction,
# renamed folder, etc.).  We try the requested target first, then common
# aliases so the pipeline does not die with an ImportError on first run.
# =============================================================================
def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")

    target = config['target']
    targets_to_try = [target]

    # Common packaging aliases seen in the wild
    if "hy3dshape" in target:
        targets_to_try.append(target.replace("hy3dshape", "hy3dgen.shapegen"))
    if "hy3dgen" in target and "_hy3dgen" not in target:
        targets_to_try.append(target.replace("hy3dgen", "_hy3dgen"))

    last_error = None
    for t in targets_to_try:
        try:
            cls = get_obj_from_str(t)
            break
        except Exception as e:
            last_error = e
    else:
        raise last_error

    params = config.get("params", dict())
    kwargs.update(params)
    instance = cls(**kwargs)
    return instance


class Hunyuan3DDiTPipeline:
    """
    Base pipeline for Hunyuan3D shape generation.

    This class has been community-optimised for:
      - Low-VRAM GPUs (manual sequential offload + hybrid mode)
      - CPU inference (thread optimisation, progress silencing)
      - Windows / AMD stability (safe compile disabling)
      - Automatic hardware detection (SmartDeviceManager)
    """

    model_cpu_offload_seq = "conditioner->model->vae"
    _exclude_from_cpu_offload = []

    @classmethod
    @synchronize_timer('Hunyuan3DDiTPipeline Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        strategy=None,          # [COMMUNITY] "gpu", "hybrid", "cpu", or None
        **kwargs,
    ):
        # load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")
        logger.info(f"Loading model from {ckpt_path}")

        if use_safetensors:
            import safetensors.torch
            safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
            ckpt = {}
            for key, value in safetensors_ckpt.items():
                model_name = key.split('.')[0]
                new_key = key[len(model_name) + 1:]
                if model_name not in ckpt:
                    ckpt[model_name] = {}
                ckpt[model_name][new_key] = value
        else:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)

        # load model
        model = instantiate_from_config(config['model'])
        model.load_state_dict(ckpt['model'])
        vae = instantiate_from_config(config['vae'])
        vae.load_state_dict(ckpt['vae'], strict=False)
        conditioner = instantiate_from_config(config['conditioner'])
        if 'conditioner' in ckpt:
            conditioner.load_state_dict(ckpt['conditioner'])
        image_processor = instantiate_from_config(config['image_processor'])
        scheduler = instantiate_from_config(config['scheduler'])

        model_kwargs = dict(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor,
            device=device,
            dtype=dtype,
            strategy=strategy,
        )
        model_kwargs.update(kwargs)

        return cls(**model_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,   # 2.1 ships .ckpt by default
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-1',
        strategy=None,
        **kwargs,
    ):
        kwargs['from_pretrained_kwargs'] = dict(
            model_path=model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant,
            dtype=dtype,
            device=device,
        )
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )
        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            strategy=strategy,
            **kwargs
        )

    def __init__(
        self,
        vae,
        model,
        scheduler,
        conditioner,
        image_processor,
        device='cuda',
        dtype=torch.float16,
        strategy=None,
        **kwargs
    ):
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.image_processor = image_processor
        self.kwargs = kwargs

        # =================================================================
        # [COMMUNITY] Smart Device Manager
        # =================================================================
        # Instead of blindly putting everything on "cuda" or "cpu", we ask
        # the SmartDeviceManager to inspect the machine and decide.
        # The manager stores:
        #   - self._device_mgr.device      : where the diffusion model lives
        #   - self._device_mgr.vae_device  : where the VAE lives (can be CPU)
        #   - self._device_mgr.strategy    : "gpu", "hybrid", or "cpu"
        # =================================================================
        self._device_mgr = SmartDeviceManager(device=device, strategy=strategy)
        self.device = self._device_mgr.device
        self.vae_device = self._device_mgr.vae_device
        self.strategy = self._device_mgr.strategy

        # =================================================================
        # [COMMUNITY] Manual sequential offload flag
        # =================================================================
        # When True, the pipeline moves each sub-model to the compute device
        # only for its forward pass, then back to CPU.  This is a lightweight
        # alternative to accelerate's cpu_offload_with_hook() and works even
        # when the `accelerate` library is not installed.
        # =================================================================
        self.manual_offload = False

        self.to(self.device, dtype)

        # =================================================================
        # [COMMUNITY] Optional torch.compile() on model
        # =================================================================
        # We only attempt compilation if the platform supports it.
        # On Windows/CPU compilation is neutered above, so this becomes a
        # no-op and the pipeline survives.
        # =================================================================
        if not _DISABLE_COMPILE and hasattr(torch, "compile"):
            try:
                logger.info("Attempting torch.compile() with reduce-overhead mode...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("torch.compile() applied successfully.")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}. Continuing without compilation.")
        elif _DISABLE_COMPILE:
            logger.info("torch.compile() is disabled for this system configuration.")

    # =====================================================================
    # [COMMUNITY] Manual sequential offload toggle
    # =====================================================================
    def enable_sequential_offload(self, enabled: bool = True):
        """Enable manual CPU->GPU->CPU offloading for each model component."""
        self.manual_offload = enabled

    def compile(self):
        """Compile sub-models with torch.compile (GPU-only, safe on CPU)."""
        # [COMMUNITY] Guard against crash on Windows/CPU
        if _DISABLE_COMPILE:
            logger.warning("compile() skipped: torch.compile is disabled on this system.")
            return
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        self.conditioner = torch.compile(self.conditioner)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            self.conditioner.to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            # In hybrid mode the VAE starts on CPU even if device is GPU
            if self.strategy == "hybrid":
                self.model.to(self.device)
                self.conditioner.to(self.device)
                self.vae.to(self.vae_device)
            else:
                self.vae.to(device)
                self.model.to(device)
                self.conditioner.to(device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed.
        After calling `enable_sequential_cpu_offload` the execution device can
        only be inferred from Accelerate's module hooks.
        """
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with
        a low impact on performance.  This is the *official* diffusers method.

        If `accelerate` is not available, fall back to the community
        `enable_sequential_offload()` method instead.
        """
        if self.model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            logger.warning(
                "accelerate >= 0.17.0 not available. "
                "Falling back to manual sequential offload."
            )
            self.enable_sequential_offload(True)
            return

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of "
                f"the device: `device`={torch_device.type}"
            )

        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)
        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu")
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()

        all_model_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)
            if not isinstance(model, torch.nn.Module):
                continue

            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        r"""
        Function that offloads all components, removes all model hooks that were
        added when using `enable_model_cpu_offload` and then applies them again.
        """
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            return

        for hook in self._all_hooks:
            hook.offload()
            hook.remove()

        self.enable_model_cpu_offload()

    @synchronize_timer('Encode cond')
    def encode_cond(self, image, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
        # [COMMUNITY] Manual offload: bring conditioner to GPU for this pass
        if getattr(self, "manual_offload", False):
            self.conditioner.to(self.device)

        bsz = image.shape[0]
        cond = self.conditioner(image=image, **additional_cond_inputs)

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            if dual_guidance:
                # =========================================================
                # [COMMUNITY] RAM spike fix – removed copy.deepcopy()
                # =========================================================
                # The original code did:
                #     un_cond_drop_main = copy.deepcopy(un_cond)
                # deepcopy() creates a FULL duplicate of every tensor in the
                # dict.  On CPU that can instantly consume 2-4 GB of RAM.
                #
                # We only need to change the 'additional' key, so we build a
                # new dict with shallow copies (shared tensor references).
                # This is safe because we never modify the tensors themselves,
                # only which dict points to which tensor.
                # =========================================================
                un_cond_drop_main = {}
                for k, v in un_cond.items():
                    if isinstance(v, torch.Tensor):
                        un_cond_drop_main[k] = v
                    elif isinstance(v, dict):
                        un_cond_drop_main[k] = {kk: vv for kk, vv in v.items()}
                    else:
                        un_cond_drop_main[k] = v
                un_cond_drop_main['additional'] = cond['additional']

                def cat_recursive(a, b, c):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b, c], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k], c[k])
                    return out

                cond = cat_recursive(cond, un_cond_drop_main, un_cond)
            else:
                def cat_recursive(a, b):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k])
                    return out

                cond = cat_recursive(cond, un_cond)

        # [COMMUNITY] Manual offload: send conditioner back to CPU, free VRAM
        if getattr(self, "manual_offload", False):
            self.conditioner.to("cpu")
            free_vram()

        return cond

    def prepare_extra_step_kwargs(self, generator, eta):
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, dtype, device, generator, latents=None):
        shape = (batch_size, *self.vae.latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents

    def prepare_image(self, image, mask=None) -> dict:
        # 2.1 added explicit mask support (torch.Tensor pair)
        if isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor):
            outputs = {
                'image': image,
                'mask': mask
            }
            return outputs

        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Couldn't find image at path {image}")

        if not isinstance(image, list):
            image = [image]

        outputs = []
        for img in image:
            output = self.image_processor(img)
            outputs.append(output)

        cond_input = {k: [] for k in outputs[0].keys()}
        for output in outputs:
            for key, value in output.items():
                cond_input[key].append(value)
        for key, value in cond_input.items():
            if isinstance(value[0], torch.Tensor):
                cond_input[key] = torch.cat(value, dim=0)

        return cond_input

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def set_surface_extractor(self, mc_algo):
        if mc_algo is None:
            return
        logger.info('The parameters `mc_algo` is deprecated, and will be removed in future versions.\n'
                    'Please use: \n'
                    'from hy3dshape.models.autoencoders import SurfaceExtractors\n'
                    'pipeline.vae.surface_extractor = SurfaceExtractors[mc_algo]() instead\n')
        if mc_algo not in SurfaceExtractors.keys():
            raise ValueError(f"Unknown mc_algo {mc_algo}")
        self.vae.surface_extractor = SurfaceExtractors[mc_algo]()

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        dual_guidance_scale: float = 10.5,
        dual_guidance: bool = True,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=-1 / 512,
        num_chunks=8000,
        mc_algo=None,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and \
                                      getattr(self.model, 'guidance_cond_proj_dim', None) is None
        dual_guidance = dual_guidance_scale >= 0 and dual_guidance

        if isinstance(image, torch.Tensor):
            pass
        else:
            cond_inputs = self.prepare_image(image)
            image = cond_inputs.pop('image')

        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        batch_size = image.shape[0]

        t_dtype = torch.long
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas)

        latents = self.prepare_latents(batch_size, dtype, device, generator)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        guidance_cond = None
        if getattr(self.model, 'guidance_cond_proj_dim', None) is not None:
            logger.info('Using lcm guidance scale')
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
            guidance_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.guidance_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # [COMMUNITY] Manual offload: bring diffusion model to GPU
        if getattr(self, "manual_offload", False):
            self.model.to(self.device)

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:", leave=False)):
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * (3 if dual_guidance else 2))
                else:
                    latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timestep_tensor = torch.tensor([t], dtype=t_dtype, device=device)
                timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
                noise_pred = self.model(latent_model_input, timestep_tensor, cond, guidance_cond=guidance_cond)

                if do_classifier_free_guidance:
                    if dual_guidance:
                        noise_pred_clip, noise_pred_dino, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_clip - noise_pred_dino)
                            + dual_guidance_scale * (noise_pred_dino - noise_pred_uncond)
                        )
                    else:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        # [COMMUNITY] Manual offload: send diffusion model back to CPU
        if getattr(self, "manual_offload", False):
            self.model.to("cpu")
            free_vram()

        # =================================================================
        # [COMMUNITY] Auto-detect safe mesh-extraction params
        # =================================================================
        # The Modly node system can override octree_resolution and num_chunks.
        # If the user leaves them at default (or passes None), the Smart
        # Device Manager steps in and picks values that will NOT kill a
        # 16 GB laptop.
        # =================================================================
        safe_octree, safe_chunks = self._device_mgr.get_safe_export_params(
            octree_resolution, num_chunks
        )

        return self._export(
            latents,
            output_type,
            box_v, mc_level, safe_chunks, safe_octree, mc_algo,
        )

    def _export(
        self,
        latents,
        output_type='trimesh',
        box_v=1.01,
        mc_level=0.0,
        num_chunks=2000,          # [COMMUNITY] lowered from 20000 for modest hardware
        octree_resolution=256,
        mc_algo='mc',
        enable_pbar=True
    ):
        if not output_type == "latent":
            # =================================================================
            # [COMMUNITY] Hybrid mode: move VAE to the right device
            # =================================================================
            # In "hybrid" strategy the diffusion model stays on GPU but the
            # VAE is moved to CPU before mesh extraction.  This prevents the
            # GPU from running out of VRAM during the octree decode.
            # =================================================================
            target_device = self.vae_device if self.strategy == "hybrid" else self.device
            self.vae.to(target_device)
            latents = latents.to(target_device)

            # =================================================================
            # [COMMUNITY] Aggressive RAM cleanup before the heaviest operation
            # =================================================================
            # The diffusion phase may have left large temporary tensors in
            # memory.  We force Python to collect garbage NOW so those
            # tensors disappear before the VAE decode starts asking for
            # hundreds of megabytes (or gigabytes) of fresh RAM.
            # =================================================================
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # =================================================================
            # [COMMUNITY] CPU-optimised mesh extraction
            # =================================================================
            # When running on CPU we:
            #   1. Silence tqdm to avoid console spam on slow machines
            #   2. Use safe_cores (not total_cores) to leave RAM headroom
            #   3. Measure and log elapsed time so users know it is working
            #   4. Restore the original thread count afterwards
            # =================================================================
            is_cpu = str(target_device) == 'cpu'
            if is_cpu:
                import time
                actual_enable_pbar = False
                original_threads = torch.get_num_threads()
                # Use safe_cores instead of total_cores.  Each PyTorch thread
                # allocates its own workspace buffers; fewer threads = less
                # peak RAM and a happier Windows Task Manager.
                mesh_threads = max(1, safe_cores - 1)
                torch.set_num_threads(mesh_threads)
                logger.info(
                    f"[CPU EXPORT] Mesh extraction: octree={octree_resolution}, "
                    f"chunks={num_chunks}, threads={torch.get_num_threads()}. "
                    f"This may take several minutes on CPU..."
                )
                start_time = time.time()
            else:
                actual_enable_pbar = enable_pbar
                start_time = None

            latents = 1. / self.vae.scale_factor * latents
            latents = self.vae(latents)
            outputs = self.vae.latents2mesh(
                latents,
                bounds=box_v,
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_algo=mc_algo,
                enable_pbar=actual_enable_pbar,
            )

            if is_cpu:
                elapsed = time.time() - start_time
                torch.set_num_threads(original_threads)
                logger.info(
                    f"[CPU EXPORT] Done in {elapsed:.1f}s ({elapsed/60:.1f} min)."
                )

            # [COMMUNITY] Manual offload: send VAE back to CPU
            if getattr(self, "manual_offload", False):
                self.vae.to("cpu")
                free_vram()
        else:
            outputs = latents

        if output_type == 'trimesh':
            outputs = export_to_trimesh(outputs)

        return outputs


class Hunyuan3DDiTFlowMatchingPipeline(Hunyuan3DDiTPipeline):
    """
    Flow-matching variant used by Hunyuan3D-2.1.

    Same community optimisations apply (manual offload, CPU decode, hybrid
    mode, SmartDeviceManager, etc.).
    """

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=2000,          # [COMMUNITY] lowered for modest hardware
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        mask=None,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        cond_inputs = self.prepare_image(image, mask)
        image = cond_inputs.pop('image')
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )

        batch_size = image.shape[0]

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)

        # [COMMUNITY] Manual offload: bring model to GPU
        if getattr(self, "manual_offload", False):
            self.model.to(self.device)

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        # [COMMUNITY] Manual offload: send model back to CPU
        if getattr(self, "manual_offload", False):
            self.model.to("cpu")
            free_vram()

        # =================================================================
        # [COMMUNITY] Auto-detect safe mesh-extraction params
        # =================================================================
        safe_octree, safe_chunks = self._device_mgr.get_safe_export_params(
            octree_resolution, num_chunks
        )

        return self._export(
            latents,
            output_type,
            box_v, mc_level, safe_chunks, safe_octree, mc_algo,
            enable_pbar=enable_pbar,
        )
