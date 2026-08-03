"""
Hunyuan3D 2 (Official Clean) — Extension setup script for Modly.
Direct HF downloader for official Hunyuan3D-2 models (.safetensors).
Auto-organizes models in Documents/Modly/models/hunyuan3d-v2.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

# ------------------------------------------------------------------ #
# Constantes y Parámetros
# ------------------------------------------------------------------ #
EXTENSION_NAME = "hunyuan3d-v2"

# Repositorio oficial de Hunyuan3D-2 en Hugging Face
HUNYUAN_REPO_ID = "tencent/Hunyuan3D-2" 
TARGET_SUBFOLDER = Path("generate") / "hunyuan3d-dit-v2-0"

# Archivos oficiales del modelo DiT V2
MODEL_FILES = [
    "config.yaml",
    "model.fp16.safetensors"
]

PACKAGES = [
    "torch",
    "torchvision",
    "Pillow",
    "numpy",
    "trimesh",
    "pymeshlab",
    "opencv-python-headless",
    "huggingface_hub",
    "diffusers>=0.31.0",
    "transformers>=4.46.0",
    "accelerate",
    "einops",
    "scipy",
    "scikit-image",
    "rembg",
    "mcubes",
]

def log(msg: str) -> None:
    print(f"[setup] {msg}")

def get_python(venv: Path) -> Path:
    is_win = platform.system() == "Windows"
    return venv / ("Scripts/python.exe" if is_win else "bin/python")

def pip(venv: Path, *args: str) -> None:
    """Ejecuta pip a través del binario de Python para evitar bloqueos de archivo en Windows."""
    python_exe = get_python(venv)
    subprocess.run([str(python_exe), "-m", "pip", *args], check=True)

def download_models(ext_dir: Path, venv: Path) -> None:
    """Descarga los modelos oficiales (.safetensors) desde Hugging Face."""
    target_dir = ext_dir / TARGET_SUBFOLDER
    target_dir.mkdir(parents=True, exist_ok=True)
    python_venv = get_python(venv)

    for file_name in MODEL_FILES:
        target_file = target_dir / file_name

        # Si el archivo existe pero pesa menos de 1 MB (y no es el yaml), se asume corrupto
        if target_file.exists():
            if target_file.stat().st_size < 1024 * 1024 and not file_name.endswith('.yaml'):
                log(f"El archivo {file_name} parece estar incompleto o corrupto. Eliminando para redescargar...")
                target_file.unlink()
            else:
                log(f"El archivo {file_name} ya está presente en: {target_file}")
                continue

        log(f"Descargando {file_name} desde Hugging Face ({HUNYUAN_REPO_ID})...")

        download_script = (
            f"from huggingface_hub import hf_hub_download\n"
            f"hf_hub_download("
            f"repo_id='{HUNYUAN_REPO_ID}', "
            f"filename='{file_name}', "
            f"subfolder='hunyuan3d-dit-v2-0', "
            f"local_dir=r'{target_dir.parent}', "
            f"local_dir_use_symlinks=False)\n"
        )

        subprocess.run([str(python_venv), "-c", download_script], check=True)
        log(f"Descarga de {file_name} completada exitosamente.")

def sync_to_modly_models(ext_dir: Path) -> None:
    """Asegura que la carpeta generate/ quede ubicada en Documents/Modly/models/hunyuan3d-v2."""
    modly_models_dir = Path.home() / "Documents" / "Modly" / "models" / EXTENSION_NAME
    local_generate = ext_dir / "generate"

    if not local_generate.exists():
        log(f"Aviso: No se encontró la carpeta 'generate' local en {local_generate}")
        return

    log(f"Configurando directorio de modelos en: {modly_models_dir}")
    modly_models_dir.mkdir(parents=True, exist_ok=True)

    target_generate = modly_models_dir / "generate"

    # Si la carpeta destino ya existe, sincronizamos el contenido
    if target_generate.exists():
        log("La carpeta 'generate' ya existe en Modly models. Actualizando contenido...")
        shutil.copytree(local_generate, target_generate, dirs_exist_ok=True)
    else:
        log("Moviendo estructura 'generate' a Modly models...")
        shutil.move(str(local_generate), str(target_generate))

    log("Directorio de modelos sincronizado con éxito.")

def setup(
    python_exe: str,
    ext_dir: Path,
    gpu_sm: int = 0,
    cuda_version: int = 0,
    torch_flavor: str = "cuda",
    accelerator: str = "",
    platform_name: str = "",
) -> None:
    venv = ext_dir / "venv"

    log(f"Iniciando configuración limpia de Hunyuan3D-2 en {ext_dir}")
    log(f"Creando entorno virtual (venv) en {venv}...")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # 1. Instalación de dependencias
    log("Instalando dependencias de Python...")
    python_venv = get_python(venv)
    subprocess.run([str(python_venv), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    pip(venv, "install", *PACKAGES)

    # 2. Enlace .pth
    log("Vinculando la extensión al entorno Python (.pth)...")
    try:
        is_win = platform.system() == "Windows"
        site_packages = venv / (
            "Lib/site-packages"
            if is_win
            else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
        site_packages.mkdir(parents=True, exist_ok=True)
        pth_file = site_packages / f"{EXTENSION_NAME}.pth"
        pth_file.write_text(str(ext_dir.resolve()), encoding="utf-8")
        log(f"Enlace .pth listo en: {pth_file.name}")
    except Exception as e:
        log(f"Aviso al crear el enlace .pth: {e}")

    # 3. Descargar modelos oficiales de Tencent
    download_models(ext_dir, venv)

    # 4. Mover/Sincronizar 'generate' a Documents/Modly/models/hunyuan3d-v2
    sync_to_modly_models(ext_dir)

    log("¡Setup finalizado con éxito! Hunyuan3D-2 oficial está listo para usar.")

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=sys.argv[1],
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
            cuda_version=int(sys.argv[4]) if len(sys.argv) >= 5 else 0,
            torch_flavor=sys.argv[5] if len(sys.argv) >= 6 else "cuda",
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe=args["python_exe"],
            ext_dir=Path(args["ext_dir"]),
            gpu_sm=args.get("gpu_sm", 0),
            cuda_version=args.get("cuda_version", 0),
            torch_flavor=args.get("torch_flavor", "cuda"),
            accelerator=args.get("accelerator", ""),
            platform_name=args.get("platform", ""),
        )
    else:
        print("Uso: python setup.py <json_args>")
        sys.exit(1)