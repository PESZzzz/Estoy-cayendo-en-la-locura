"""
Hunyuan3D 2 Mini (GGUF) — Extension setup script for Modly.
"""

import io
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

# ------------------------------------------------------------------ #
# Constantes y Parámetros
# ------------------------------------------------------------------ #
EXTENSION_NAME = "hunyuan3d-v2-gguf"
HUNYUAN_GITHUB_ZIP = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"

FILES_TO_REPLACE = [
    "pipelines.py",
    "surface_extractors.py",
    "volume_decoders.py",
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
    "gguf",
    "ninja",
    "rembg",
    "onnxruntime",
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


def download_hy3dgen_source(generate_dir: Path) -> None:
    """Descarga y extrae la estructura base de hy3dgen dentro de generate/_hy3dgen."""
    target_hy3dgen = generate_dir / "_hy3dgen"
    if (target_hy3dgen / "hy3dgen").exists():
        log("El módulo _hy3dgen ya existe dentro de 'generate'.")
        return

    log("Descargando código fuente de hy3dgen desde GitHub...")
    target_hy3dgen.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(HUNYUAN_GITHUB_ZIP, timeout=180) as resp:
        data = resp.read()

    log("Extrayendo hy3dgen dentro de generate/_hy3dgen...")
    prefix = "Hunyuan3D-2-main/hy3dgen/"
    strip = "Hunyuan3D-2-main/"

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if not member.startswith(prefix):
                continue
            rel = member[len(strip):]
            target = target_hy3dgen / rel
            if member.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(member))

    log(f"hy3dgen configurado correctamente en: {target_hy3dgen}")


def replace_custom_files(ext_dir: Path, generate_dir: Path) -> None:
    """Copia y reemplaza los parches personalizados en generate/_hy3dgen/hy3dgen/shapegen."""
    target_dir = generate_dir / "_hy3dgen" / "hy3dgen" / "shapegen"
    
    if not target_dir.exists():
        log(f"Aviso: La ruta destino {target_dir} aún no existe.")
        return

    log("Aplicando parches personalizados GGUF en hy3dgen/shapegen...")
    for filename in FILES_TO_REPLACE:
        src_file = ext_dir / filename
        dst_file = target_dir / filename

        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            log(f"Reemplazado: {filename} -> {dst_file}")
        else:
            log(f"Aviso: No se encontró el parche local {src_file}")


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

    log(f"Iniciando configuración en {ext_dir}")
    log(f"Creando entorno virtual (venv) en {venv}...")
    subprocess.run([python_exe, "-m", "venv", str(venv)], check=True)

    # ------------------------------------------------------------------ #
    # Instalación de dependencias de Python
    # ------------------------------------------------------------------ #
    log("Instalando dependencias necesarias (PyTorch, Pillow, GGUF, etc.)...")
    
    # Usamos python -m pip para evitar que Windows bloquee pip.exe al actualizar
    python_venv = get_python(venv)
    subprocess.run([str(python_venv), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    pip(venv, "install", *PACKAGES)

    # ------------------------------------------------------------------ #
    # Enlace .pth para resolver importaciones
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Preparación de estructura hy3dgen local
    # ------------------------------------------------------------------ #
    generate_dir = ext_dir / "generate"
    generate_dir.mkdir(parents=True, exist_ok=True)

    download_hy3dgen_source(generate_dir)
    replace_custom_files(ext_dir, generate_dir)

    log("¡Setup finalizado con éxito! Entorno listo para Modly.")


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
            gpu_sm=int(args.get("gpu_sm", 0)),
            cuda_version=int(args.get("cuda_version", 0)),
            torch_flavor=args.get("torch_flavor", "cuda"),
            accelerator=args.get("accelerator", ""),
            platform_name=args.get("platform", ""),
        )
    else:
        print("Uso: python setup.py <json_args>")
        sys.exit(1)
