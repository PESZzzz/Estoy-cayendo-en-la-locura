# Hunyuan 3D - Optimized Pipeline for Modly / GGUF Integrations
import copy
import importlib
import inspect
import os
from typing import List, Optional, Union

import numpy as np
import torch
import trimesh
import yaml
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_accelerate_available
from tqdm import tqdm

from .models.autoencoders import ShapeVAE, SurfaceExtractors
from .utils import logger, synchronize_timer, smart_load_model


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support custom timesteps.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"Scheduler {scheduler.__class__} does not support custom sigmas.")
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
                outputs.append(trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f))
        return outputs
    else:
        return trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)


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
    except Exception:
        target = config['target'].replace("hy3dshape", "hy3dgen.shapegen")
        cls = get_obj_from_str(target)
    params = config.get("params", dict())
    kwargs.update(params)
    return cls(**kwargs)


def _load_gguf_to_state_dict(gguf_path: str, strip_prefix: str = "", dtype=torch.float16):
    """Carga de tensores desde un archivo GGUF mapeándolos directamente a PyTorch."""
    import gguf
    reader = gguf.GGUFReader(gguf_path)
    state_dict = {}
    for tensor in reader.tensors:
        t_name = tensor.name
        if strip_prefix and t_name.startswith(strip_prefix):
            t_name = t_name[len(strip_prefix):]
        
        t_data = torch.from_numpy(tensor.data)
        if t_data.is_floating_point():
            t_data = t_data.to(dtype)
        state_dict[t_name] = t_data
    return state_dict


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

        model_dir = os.path.dirname(ckpt_path) if os.path.isfile(ckpt_path) else ckpt_path
        
        # 0. Checkpoint base opcional
        ckpt = {}
        if os.path.isfile(ckpt_path) and os.path.exists(ckpt_path):
            logger.info(f"[MODLY] Leyendo checkpoint base: {ckpt_path}")
            try:
                if ckpt_path.endswith('.safetensors'):
                    import safetensors.torch
                    safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
                    for key, value in safetensors_ckpt.items():
                        model_name = key.split('.')[0]
                        new_key = key[len(model_name) + 1:]
                        if model_name not in ckpt:
                            ckpt[model_name] = {}
                        ckpt[model_name][new_key] = value
                else:
                    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            except Exception as e:
                logger.info(f"[MODLY] Carga unificada omitida ({e}). Usando módulos independientes.")

        # 1. Instanciación e Inyección del DiT (GGUF Principal)
        model = instantiate_from_config(config['model'])
        dit_gguf_path = None
        
        if os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.gguf') and not file.startswith('pig'):
                    dit_gguf_path = os.path.join(model_dir, file)
                    break

        if dit_gguf_path and os.path.exists(dit_gguf_path):
            logger.info(f"[MODLY-GGUF] Cargando DiT (Core MM-DiT) desde: {dit_gguf_path}")
            try:
                gguf_sd = _load_gguf_to_state_dict(dit_gguf_path, strip_prefix="model.", dtype=dtype)
                missing, unexpected = model.load_state_dict(gguf_sd, strict=False)
                logger.info(f"[MODLY-GGUF] DiT cargado con éxito. (Faltantes: {len(missing)})")
            except Exception as e:
                logger.warning(f"[MODLY-GGUF] Error en GGUF DiT ({e}). Intentando fallback...")
                if 'model' in ckpt:
                    model.load_state_dict(ckpt['model'], strict=False)
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'], strict=False)

        # 2. Instanciación e Inyección del VAE (Pig 3D VAE GGUF)
        vae = instantiate_from_config(config['vae'])
        vae_gguf_path = os.path.join(model_dir, "pig_3d_vae_fp32-f16.gguf")
        
        if os.path.exists(vae_gguf_path):
            logger.info(f"[MODLY-GGUF] Cargando Pig VAE (Latent Dim 64) desde: {vae_gguf_path}")
            try:
                vae_sd = _load_gguf_to_state_dict(vae_gguf_path, strip_prefix="vae.", dtype=dtype)
                vae.load_state_dict(vae_sd, strict=False)
                logger.info("[MODLY-GGUF] Pig VAE cargado exitosamente.")
            except Exception as e:
                logger.warning(f"[MODLY-GGUF] Error en Pig VAE: {e}")
                if 'vae' in ckpt:
                    vae.load_state_dict(ckpt['vae'], strict=False)
        elif 'vae' in ckpt:
            vae.load_state_dict(ckpt['vae'], strict=False)

        # 3. Instanciación e Inyección del Vision Encoder (Conditioner)
        conditioner = instantiate_from_config(config['conditioner'])
        vision_path = os.path.join(model_dir, "hy-3d-vision.safetensors")
        
        if os.path.exists(vision_path):
            logger.info(f"[MODLY-GGUF] Cargando Vision Encoder desde: {vision_path}")
            try:
                import safetensors.torch
                vision_weights = safetensors.torch.load_file(vision_path, device='cpu')
                conditioner.load_state_dict(vision_weights, strict=False)
                logger.info("[MODLY-GGUF] Vision Encoder cargado exitosamente.")
            except Exception as e:
                logger.warning(f"[MODLY-GGUF] Error en Vision Encoder: {e}")
                if 'conditioner' in ckpt:
                    conditioner.load_state_dict(ckpt['conditioner'], strict=False)
        elif 'conditioner' in ckpt:
            conditioner.load_state_dict(ckpt['conditioner'], strict=False)

        # 4. Procesadores y Scheduler
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

        # 1. Resolver directorio del modelo
        if os.path.exists(model_path):
            target_dir = os.path.join(model_path, subfolder) if subfolder and not model_path.endswith(subfolder) else model_path
        else:
            config_path, target_dir = smart_load_model(
                model_path,
                subfolder=subfolder,
                use_safetensors=use_safetensors,
                variant=variant
            )

        # 2. Resolver Configuración
        config_path = os.path.join(target_dir, "config.yaml") if os.path.isdir(target_dir) else target_dir

        # 3. Buscar archivo de pesos principal (.safetensors / .ckpt / .gguf)
        ckpt_path = None
        if os.path.isdir(target_dir):
            valid_exts = ('.safetensors', '.ckpt', '.gguf', '.pt')
            for file in os.listdir(target_dir):
                if file.endswith(valid_exts):
                    ckpt_path = os.path.join(target_dir, file)
                    logger.info(f"[MODLY] Archivo de pesos detectado: {ckpt_path}")
                    break
        else:
            ckpt_path = target_dir

        if not ckpt_path or not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"No se encontró un archivo de pesos válido en: {target_dir}")

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
        self.to(device, dtype)

    def compile(self):
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

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        if not is_accelerate_available():
            raise ImportError("`enable_model_cpu_offload` requiere la librería `accelerate`.")

        from accelerate import cpu_offload_with_hook
        torch_device = torch.device(device)
        self._offload_gpu_id = gpu_id or torch_device.index or 0
        target_device = torch.device(f"{torch_device.type}:{self._offload_gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}
        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            mod = all_components.pop(model_str, None)
            if isinstance(mod, torch.nn.Module):
                _, hook = cpu_offload_with_hook(mod, target_device, prev_module_hook=hook)
                self._all_hooks.append(hook)

    @property
    def components(self):
        return {
            "vae": self.vae,
            "model": self.model,
            "conditioner": self.conditioner,
            "image_processor": self.image_processor,
            "scheduler": self.scheduler,
        }

    @synchronize_timer('Encode cond')
    def encode_cond(self, image, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
        bsz = image.shape[0]
        cond = self.conditioner(image=image, **additional_cond_inputs)

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            def cat_recursive(a, b):
                if isinstance(a, torch.Tensor):
                    return torch.cat([a, b], dim=0).to(self.dtype)
                out = {}
                for k in a.keys():
                    out[k] = cat_recursive(a[k], b[k])
                return out

            cond = cat_recursive(cond, un_cond)
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
            latents = latents.to(device=device, dtype=dtype)

        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents

    def prepare_image(self, image) -> dict:
        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Imagen no encontrada en la ruta: {image}")

        if not isinstance(image, list):
            image = [image]

        outputs = [self.image_processor(img) for img in image]
        cond_input = {k: [] for k in outputs[0].keys()}
        for output in outputs:
            for key, value in output.items():
                cond_input[key].append(value)
        for key, value in cond_input.items():
            if isinstance(value[0], torch.Tensor):
                cond_input[key] = torch.cat(value, dim=0)

        return cond_input

    def set_surface_extractor(self, mc_algo):
        if mc_algo is None:
            return
        if mc_algo not in SurfaceExtractors.keys():
            raise ValueError(f"Extractor desconocido {mc_algo}")
        self.vae.surface_extractor = SurfaceExtractors[mc_algo]()

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image] = None,
        num_inference_steps: int = 30,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        num_chunks=65536,
        mc_algo=None,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        **kwargs,
    ) -> List[trimesh.Trimesh]:

        self.set_surface_extractor(mc_algo)
        device = self.device
        dtype = self.dtype

        do_classifier_free_guidance = guidance_scale >= 0
        cond_inputs = self.prepare_image(image)
        img_tensor = cond_inputs.pop('image')
        
        cond = self.encode_cond(
            image=img_tensor,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        batch_size = img_tensor.shape[0]

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        latents = self.prepare_latents(batch_size, dtype, device, generator)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Sampling 3D Mesh")):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                timestep_tensor = torch.tensor([t], dtype=torch.long, device=device).expand(latent_model_input.shape[0])
                noise_pred = self.model(latent_model_input, timestep_tensor, cond)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = outputs.prev_sample

        return self._export(
            latents, output_type, box_v, mc_level, num_chunks, octree_resolution, mc_algo
        )

    def _export(
        self,
        latents,
        output_type='trimesh',
        box_v=1.01,
        mc_level=0.0,
        num_chunks=65536,
        octree_resolution=384,
        mc_algo='mc',
        enable_pbar=True
    ):
        if output_type != "latent":
            latents = 1. / getattr(self.vae, 'scale_factor', 1.0) * latents
            latents = self.vae(latents)
            outputs = self.vae.latents2mesh(
                latents,
                bounds=box_v,
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_algo=mc_algo,
                enable_pbar=enable_pbar,
            )
        else:
            outputs = latents

        if output_type == 'trimesh':
            outputs = export_to_trimesh(outputs)

        return outputs


class Hunyuan3DDiTFlowMatchingPipeline(Hunyuan3DDiTPipeline):

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image] = None,
        num_inference_steps: int = 30,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        num_chunks=65536,
        mc_algo=None,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        **kwargs,
    ) -> List[trimesh.Trimesh]:

        self.set_surface_extractor(mc_algo)
        device = self.device
        dtype = self.dtype

        do_classifier_free_guidance = guidance_scale >= 0
        cond_inputs = self.prepare_image(image)
        img_tensor = cond_inputs.pop('image')

        cond = self.encode_cond(
            image=img_tensor,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        batch_size = img_tensor.shape[0]

        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, sigmas=sigmas
        )
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        with synchronize_timer('Flow Matching Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Flow Sampling 3D")):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype) / self.scheduler.config.num_train_timesteps

                noise_pred = self.model(latent_model_input, timestep, cond)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

        return self._export(
            latents, output_type, box_v, mc_level, num_chunks, octree_resolution, mc_algo, enable_pbar=enable_pbar
        )
