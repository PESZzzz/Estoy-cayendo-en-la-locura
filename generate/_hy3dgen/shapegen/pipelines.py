# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

import copy
import gc
import importlib
import inspect
import os
import shutil
import sys
import warnings
from typing import List, Optional, Union

import numpy as np
import torch

# ---------- Universal CPU/GPU Optimization ----------
total_cores = os.cpu_count() or 4
# Reserve 25% for OS and other apps (was 20%, increased for stability)
safe_cores = max(1, int(total_cores * 0.75))

torch.set_num_threads(safe_cores)
torch.set_num_interop_threads(max(1, int(safe_cores // 2)))
os.environ["OMP_NUM_THREADS"] = str(safe_cores)
os.environ["MKL_NUM_THREADS"] = str(safe_cores)

# Disable aggressive optimizations that crash on consumer hardware
torch.backends.mkldnn.enabled = True
torch.set_float32_matmul_precision("medium")

# Disable torch.compile/dynamo globally for CPU/Windows without MSVC
# This prevents "cl is not found" errors on AMD/Intel laptops

def _find_msvc_cl():
    """Search for cl.exe in common Visual Studio install paths."""
    if shutil.which("cl"):
        return True
    # Common VS Build Tools paths
    program_files = [r"C:\Program Files", r"C:\Program Files (x86)"]
    years = ["2022", "2019", "2017"]
    for pf in program_files:
        for year in years:
            base = os.path.join(pf, f"Microsoft Visual Studio\{year}\BuildTools\VC\Tools\MSVC")
            if os.path.isdir(base):
                # Find latest version subfolder
                try:
                    versions = sorted(os.listdir(base), reverse=True)
                    for ver in versions:
                        cl_path = os.path.join(base, ver, "bin", "Hostx64", "x64", "cl.exe")
                        if os.path.isfile(cl_path):
                            return True
                except Exception:
                    pass
    return False

_DISABLE_COMPILE = False
if sys.platform == "win32" and not torch.cuda.is_available():
    if not _find_msvc_cl():
        _DISABLE_COMPILE = True
        warnings.warn(
            "MSVC compiler (cl.exe) not found. Disabling torch.compile() "
            "for CPU inference on Windows. Install Visual Studio Build Tools "
            "with 'Desktop development with C++' workload for faster inference.",
            UserWarning,
            stacklevel=2,
        )

if _DISABLE_COMPILE:
    # Disable dynamo completely to avoid lazy compilation crashes
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.reset()
    # Monkey-patch torch.compile to be a no-op on CPU/Windows
    _original_compile = torch.compile
    def _noop_compile(model, *args, **kwargs):
        warnings.warn(
            "torch.compile() is disabled on this system (CPU/Windows without MSVC). "
            "Inference will work but may be slower.",
            UserWarning,
            stacklevel=3,
        )
        return model
    torch.compile = _noop_compile
# ----------------------------------------------------

import trimesh
import yaml
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_accelerate_version, is_accelerate_available
from tqdm import tqdm

from .models.autoencoders import ShapeVAE
from .models.autoencoders import SurfaceExtractors
from .utils import logger, synchronize_timer, smart_load_model


def free_vram():
    """Free residual GPU memory to prevent hangs on laptops and AMD."""
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


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    try:
        target = config['target']
        cls = get_obj_from_str(target)
    except Exception as e:
        target = config['target'].replace("hy3dshape", "hy3dgen.shapegen")
        cls = get_obj_from_str(target)
    params = config.get("params", dict())
    kwargs.update(params)
    instance = cls(**kwargs)
    return instance


class Hunyuan3DDiTPipeline:
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
        **kwargs,
    ):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

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
        )
        model_kwargs.update(kwargs)

        return cls(**model_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=True,
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-0',
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
        **kwargs
    ):
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.image_processor = image_processor
        self.kwargs = kwargs

        self.manual_offload = False
        self.to(device, dtype)

        # Only attempt torch.compile if not disabled for this system
        if not _DISABLE_COMPILE and hasattr(torch, "compile"):
            try:
                logger.info("Attempting torch.compile() with reduce-overhead mode...")
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("torch.compile() applied successfully.")
            except Exception as e:
                logger.warning(f"torch.compile() failed: {e}. Continuing without compilation.")
        elif _DISABLE_COMPILE:
            logger.info("torch.compile() is disabled for this system configuration.")

    def enable_sequential_offload(self, enabled: bool = True):
        self.manual_offload = enabled

    def compile(self):
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
            self.vae.to(device)
            self.model.to(device)
            self.conditioner.to(device)

    @synchronize_timer('Encode cond')
    def encode_cond(self, image, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
        if getattr(self, "manual_offload", False):
            self.conditioner.to(self.device)

        bsz = image.shape[0]
        cond = self.conditioner(image=image, **additional_cond_inputs)

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            if dual_guidance:
                un_cond_drop_main = copy.deepcopy(un_cond)
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
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents

    def prepare_image(self, image) -> dict:
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
        w = w * 1000.0
        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return emb

    def set_surface_extractor(self, mc_algo):
        if mc_algo is None:
            return
        if mc_algo not in SurfaceExtractors.keys():
            raise ValueError(f"Unknown mc_algo {mc_algo}")
        self.vae.surface_extractor = SurfaceExtractors[mc_algo]()

    @torch.inference_mode()
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
        num_chunks=2000,
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
        do_classifier_free_guidance = guidance_scale >= 0 and getattr(self.model, 'guidance_cond_proj_dim', None) is None
        dual_guidance = dual_guidance_scale >= 0 and dual_guidance

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
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, timesteps, sigmas)
        latents = self.prepare_latents(batch_size, dtype, device, generator)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        guidance_cond = None
        if getattr(self.model, 'guidance_cond_proj_dim', None) is not None:
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
            guidance_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.guidance_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        if getattr(self, "manual_offload", False):
            self.model.to(device)

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:", leave=False)):
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * (3 if dual_guidance else 2))
                else:
                    latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timestep_tensor = t.expand(latent_model_input.shape[0]).to(dtype=t_dtype)
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

        if getattr(self, "manual_offload", False):
            self.model.to("cpu")
            free_vram()

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
        )

    def _export(
        self,
        latents,
        output_type='trimesh',
        box_v=1.01,
        mc_level=0.0,
        num_chunks=2000,
        octree_resolution=256,
        mc_algo='mc',
        enable_pbar=True
    ):
        if not output_type == "latent":
            if getattr(self, "manual_offload", False):
                self.vae.to(self.device)

            # ----- CPU-specific optimizations for mesh extraction -----
            is_cpu = str(self.device) == 'cpu'
            if is_cpu:
                # Disable progress bar on CPU to prevent log spam
                # (latents2mesh can generate 100k+ tqdm lines)
                actual_enable_pbar = False
                # Temporarily use ALL cores for mesh extraction
                import time
                original_threads = torch.get_num_threads()
                torch.set_num_threads(total_cores)
                logger.info(
                    f"[CPU EXPORT] Starting mesh extraction: "
                    f"octree={octree_resolution}, chunks={num_chunks}, "
                    f"threads={torch.get_num_threads()}, mc_algo={mc_algo}. "
                    f"This may take several minutes on CPU..."
                )
                start_time = time.time()
            else:
                actual_enable_pbar = enable_pbar
                start_time = None
            # ----------------------------------------------------------

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

            # ----- Restore settings and log completion -----
            if is_cpu:
                elapsed = time.time() - start_time
                torch.set_num_threads(original_threads)
                logger.info(
                    f"[CPU EXPORT] Mesh extraction completed in {elapsed:.1f}s "
                    f"({elapsed/60:.1f} min). Restored threads to {original_threads}."
                )
            # ------------------------------------------------

            if getattr(self, "manual_offload", False):
                self.vae.to("cpu")
                free_vram()
        else:
            outputs = latents

        if output_type == 'trimesh':
            outputs = export_to_trimesh(outputs)

        return outputs


class Hunyuan3DDiTFlowMatchingPipeline(Hunyuan3DDiTPipeline):

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, dict, List[dict]] = None,
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
        num_chunks=2000,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)
        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and self.model.guidance_embed is True
        )

        cond_inputs = self.prepare_image(image)
        image = cond_inputs.pop('image')
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        batch_size = image.shape[0]

        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, 'guidance_embed') and self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)

        if getattr(self, "manual_offload", False):
            self.model.to(device)

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                timestep = t.expand(latent_model_input.shape[0]).to(
                    latents.dtype) / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        if getattr(self, "manual_offload", False):
            self.model.to("cpu")
            free_vram()

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )
