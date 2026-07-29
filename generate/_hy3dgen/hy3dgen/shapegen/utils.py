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

import logging
import os
from functools import wraps

import torch


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


logger = get_logger('hy3dgen.shapgen')


class synchronize_timer:
    """ Synchronized timer to count the inference time of `nn.Module.forward`.

        Supports both context manager and decorator usage.

        Example as context manager:
        ```python
        with synchronize_timer('name') as t:
            run()
        ```

        Example as decorator:
        ```python
        @synchronize_timer('Export to trimesh')
        def export_to_trimesh(mesh_output):
            pass
        ```
    """

    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        """Context manager entry: start timing."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
            return lambda: self.time

    def __exit__(self, exc_type, exc_value, exc_tb):
        """Context manager exit: stop timing and log results."""
        if os.environ.get('HY3DGEN_DEBUG', '0') == '1':
            self.end.record()
            torch.cuda.synchronize()
            self.time = self.start.elapsed_time(self.end)
            if self.name is not None:
                logger.info(f'{self.name} takes {self.time} ms')

    def __call__(self, func):
        """Decorator: wrap the function to time its execution."""

        @wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result

        return wrapper


def smart_load_model(
    model_path,
    subfolder,
    use_safetensors,
    variant,
):
    original_model_path = model_path
    
    # 1. Resolvemos la ruta base si es una ruta local existente
    if os.path.exists(model_path):
        target_path = os.path.join(model_path, subfolder) if subfolder else model_path
        if os.path.exists(target_path):
            model_path = target_path
    else:
        base_dir = os.environ.get('HY3DGEN_MODELS', '~/.cache/hy3dgen')
        model_path = os.path.expanduser(os.path.join(base_dir, model_path, subfolder))

    # 2. Si no existe aún, intenta con HuggingFace
    if not os.path.exists(model_path):
        logger.info('Model path not exists, try to download from huggingface')
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(
                repo_id=original_model_path,
                allow_patterns=[f"{subfolder}/*"],
            )
            model_path = os.path.join(path, subfolder)
        except Exception as e:
            raise FileNotFoundError(f"Model path {original_model_path} not found") from e

    # 3. Definir nombres de archivo esperados
    ext = 'safetensors' if use_safetensors else 'ckpt'
    variant_str = f'.{variant}' if variant else ''
    
    config_path = os.path.join(model_path, 'config.yaml')
    
    # Búsqueda exhaustiva del archivo de pesos
    possible_ckpts = [
        os.path.join(model_path, f'model{variant_str}.{ext}'),
        os.path.join(model_path, f'model.{ext}'),
    ]
    
    ckpt_path = None
    for p in possible_ckpts:
        if os.path.exists(p):
            ckpt_path = p
            break
            
    # Si aún no lo encuentra, busca CUALQUIER archivo .safetensors/.ckpt en esa carpeta
    if not ckpt_path and os.path.exists(model_path):
        for file in os.listdir(model_path):
            if file.endswith(f'.{ext}'):
                ckpt_path = os.path.join(model_path, file)
                break

    if not ckpt_path:
        raise FileNotFoundError(f"No se encontró ningún archivo de pesos (.{ext}) en: {model_path}")

    return config_path, ckpt_path