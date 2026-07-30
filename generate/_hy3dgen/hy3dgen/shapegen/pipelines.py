# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

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
from diffusers.utils.import_utils import is_accelerate_version, is_accelerate_available
from tqdm import tqdm

from .models.autoencoders import ShapeVAE
from .models.autoencoders import SurfaceExtractors
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

        model_dir = os.path.dirname(ckpt_path) if os.path.isfile(ckpt_path) else ckpt_path
        
        # 0. Intentar carga del checkpoint base (si existe)
        ckpt = {}
        if os.path.isfile(ckpt_path) and os.path.exists(ckpt_path):
            logger.info(f"[MODLY] Intentando leer checkpoint base desde: {ckpt_path}")
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
                logger.info(f"[MODLY] No se pudo leer {ckpt_path} como paquete unificado ({e}). Se usarán archivos modulares.")

        # 1. Instanciación e Inyección del DiT (GGUF)
        model = instantiate_from_config(config['model'])
        dit_gguf_path = None
        
        if os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.gguf') and not file.startswith('pig'):
                    dit_gguf_path = os.path.join(model_dir, file)
                    break

        if dit_gguf_path and os.path.exists(dit_gguf_path):
            logger.info(f"\n{'='*60}\n[MODLY-GGUF] 🚀 Cargando DiT desde: {dit_gguf_path}")
            try:
                import gguf
                reader = gguf.GGUFReader(dit_gguf_path)
                gguf_state_dict = {}
                for tensor in reader.tensors:
                    t_name = tensor.name
                    if t_name.startswith("model."):
                        t_name = t_name[6:]
                    t_data = torch.from_numpy(tensor.data)
                    if t_data.is_floating_point():
                        t_data = t_data.to(dtype)
                    gguf_state_dict[t_name] = t_data

                missing, unexpected = model.load_state_dict(gguf_state_dict, strict=False)
                logger.info(f"[MODLY-GGUF] ✅ DiT cargado con éxito. (Faltantes: {len(missing)})\n{'='*60}\n")
            except Exception as e:
                logger.warning(f"[MODLY-GGUF] ❌ Error al leer GGUF ({e}).")
                if 'model' in ckpt:
                    model.load_state_dict(ckpt['model'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])

        # 2. Instanciación e Inyección del VAE (Pig VAE GGUF o base)
        vae = instantiate_from_config(config['vae'])
        vae_gguf_path = os.path.join(model_dir, "pig_3d_vae_fp32-f16.gguf")
        
        if os.path.exists(vae_gguf_path):
            logger.info(f"[MODLY-GGUF] 🐷 Cargando Pig VAE desde: {vae_gguf_path}")
            try:
                import gguf
                reader_vae = gguf.GGUFReader(vae_gguf_path)
                vae_state_dict = {}
                for tensor in reader_vae.tensors:
                    t_name = tensor.name
                    # Parche: Remueve prefijos comunes de GGUF VAEs
                    for prefix in ["vae.", "first_stage_model.", "decoder."]:
                        if t_name.startswith(prefix):
                            t_name = t_name[len(prefix):]
                    t_data = torch.from_numpy(tensor.data)
                    if t_data.is_floating_point():
                        t_data = t_data.to(dtype)
                    vae_state_dict[t_name] = t_data
                vae.load_state_dict(vae_state_dict, strict=False)
                logger.info("[MODLY-GGUF] ✅ Pig VAE cargado exitosamente.")
            except Exception as e:
                logger.warning(f"[MODLY-GGUF] ⚠️ Error al leer Pig VAE: {e}")
                if 'vae' in ckpt:
                    vae.load_state_dict(ckpt['vae'], strict=False)
        elif 'vae' in ckpt:
            vae.load_state_dict(ckpt['vae'], strict=False)

        # 3. Instanciación e Inyección del Conditioner / Vision Encoder
        conditioner = instantiate_from_config(config['conditioner'])
        
        # Buscar variantes de nombres para el Vision Encoder (.safetensors)
        vision_path = None
        if os.path.isdir(model_dir):
            for file in os.listdir(model_dir):
                if file.endswith('.safetensors') and ('vision' in file.lower() or 'mv' in file.lower()):
                    vision_path = os.path.join(model_dir, file)
                    break
        
        if vision_path and os.path.exists(vision_path):
            logger.info(f"[MODLY-GGUF] 👁️ Cargando Vision Encoder desde: {vision_path}")
            try:
                import safetensors.torch
                vision_weights = safetensors.torch.load_file(vision_path, device='cpu')
                
                # Filtrar solo claves pertenecientes al conditioner/vision si vienen de un ckpt completo
                cond_weights = {}
                for k, v in vision_weights.items():
                    if k.startswith("conditioner."):
                        cond_weights[k[12:]] = v
                    elif k.startswith("vision_model."):
                        cond_weights[k] = v
                    else:
                        cond_weights[k] = v
                        
                conditioner.load_state_dict(cond_weights, strict=False)
                logger.info("[MODLY-GGUF] ✅ Vision Encoder cargado exitosamente.")
            except Exception as e:
                logger.warning(f"[MODLY-GGUF] ⚠️ Error al leer Vision Encoder: {e}")
                if 'conditioner' in ckpt:
                    conditioner.load_state_dict(ckpt['conditioner'])
        elif 'conditioner' in ckpt:
            conditioner.load_state_dict(ckpt['conditioner'])

        # 4. Procesadores finales
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

        # 1. Resolver la carpeta contenedora del modelo local
        if os.path.exists(model_path):
            target_dir = os.path.join(model_path, subfolder) if subfolder and not model_path.endswith(subfolder) else model_path
        else:
            config_path, target_dir = smart_load_model(
                model_path,
                subfolder=subfolder,
                use_safetensors=use_safetensors,
                variant=variant
            )

        # 2. Configuración
        config_path = os.path.join(target_dir, "config.yaml") if os.path.isdir(target_dir) else target_dir

        # 3. Buscar el ARCHIVO real de pesos (.safetensors / .ckpt)
        ext = 'safetensors' if use_safetensors else 'ckpt'
        ckpt_path = None

        if os.path.isdir(target_dir):
            candidates = [
                os.path.join(target_dir, f"model.{variant}.{ext}" if variant else f"model.{ext}"),
                os.path.join(target_dir, f"model.{ext}"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    ckpt_path = cand
                    break

            # 3. Buscar el ARCHIVO real de pesos (.safetensors / .ckpt / .gguf)
        ext = 'safetensors' if use_safetensors else 'ckpt'
        ckpt_path = None

        if os.path.isdir(target_dir):
            # Escanear la carpeta en busca de cualquier archivo que termine con la extensión requerida
            for file in os.listdir(target_dir):
                if file.endswith(f".{ext}"):
                    ckpt_path = os.path.join(target_dir, file)
                    logger.info(f"[MODLY] ¡Archivo de pesos encontrado!: {ckpt_path}")
                    break
        else:
            ckpt_path = target_dir

        if not ckpt_path or not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No se encontró un archivo de pesos válidos (.{ext}) dentro de: {target_dir}")
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

    def enable_flashvdm(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='mc',
        replace_vae=True,
    ):
        if enabled:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            turbo_vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini-turbo'),
            }
            model_name = os.path.basename(os.path.normpath(model_path))
            if replace_vae and model_name in turbo_vae_mapping:
                model_path, subfolder = turbo_vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(
                    model_path, subfolder=subfolder,
                    use_safetensors=self.kwargs['from_pretrained_kwargs']['use_safetensors'],
                    device=self.device,
                )
            self.vae.enable_flashvdm_decoder(
                enabled=enabled,
                adaptive_kv_selection=adaptive_kv_selection,
                topk_mode=topk_mode,
                mc_algo=mc_algo
            )
        else:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini'),
            }
            model_name = os.path.basename(os.path.normpath(model_path))
            if model_name in vae_mapping:
                model_path, subfolder = vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(model_path, subfolder=subfolder)
            self.vae.enable_flashvdm_decoder(enabled=False)

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

    @property
    def _execution_device(self):
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
        if self.model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
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
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            return

        for hook in self._all_hooks:
            hook.offload()
            hook.remove()

        self.enable_model_cpu_offload()

    @synchronize_timer('Encode cond')
    def encode_cond(self, image, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
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

    def prepare_image(self, image) -> dict:
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
                    'from hy3dgen.shapegen.models.autoencoders import SurfaceExtractors\n'
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
        num_chunks=65536,  # Optimizado a batch masivo por defecto
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
        num_chunks=65536,  # Optimizado a batch masivo por defecto
        octree_resolution=256,
        mc_algo='mc',
        enable_pbar=True
    ):
        if not output_type == "latent":
            latents = 1. / self.vae.scale_factor * latents
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
        num_chunks=65536,  # Optimizado a batch masivo por defecto
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
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
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
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)

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

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )
