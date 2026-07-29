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

# Repositorios y Rutas
REPO_URL = "https://github.com/PESZzzz/Estoy-cayendo-en-la-locura"
EXTENSION_NAME = "hunyuan3d-v2-gguf"
HUNYUAN_HF_REPO = "tencent/Hunyuan3D-2"
GGUF_REPO = "calcuis/hy3d-gguf"
GGUF_FILE = "hy-3d_fp32-q4_k_m.gguf"

HUNYUAN_GITHUB_ZIP = "https://github.com/Tencent/Hunyuan3D-2/archive/refs/heads/main.zip"

# Archivos personalizados que deben reemplazar los originales del motor en hy3dgen/shapegen
FILES_TO_REPLACE = [
    "pipelines.py",
    "surface_extractors.py",
    "volume_decoders.py",
    "utils.py",  # <--- Agregado utils.py
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

BLACKLIST_DIRS = [
    "hunyuan3d-dit-v2-0-fast",
    "hunyuan3d-dit-v2-0-turbo",
    "hunyuan3d-vae-v2-0-turbo",
    "hunyuan3d-vae-v2-0-withencoder",
    "samples",
]


def log(msg: str):
    print(f"[setup] {msg}")


def get_modly_paths():
    modly = Path.home() / "Documents" / "Modly"
    return {
        "root": modly,
        "extensions": modly / "extensions",
        "models": modly / "models",
    }


def pip(venv: Path, *args: str) -> None:
    is_win = platform.system() == "Windows"
    pip_exe = venv / ("Scripts/pip.exe" if is_win else "bin/pip")
    subprocess.run([str(pip_exe), *args], check=True)


def get_python(venv: Path) -> Path:
    is_win = platform.system() == "Windows"
    return venv / ("Scripts/python.exe" if is_win else "bin/python")


def clone_repository(destination: Path):
    if destination.exists():
        log("El repositorio de la extensión ya existe.")
        return

    log("Clonando el repositorio de la extensión...")
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", REPO_URL, str(destination)], check=True)


def create_venv(ext_dir: Path) -> Path:
    venv = ext_dir / "venv"
    if venv.exists():
        log("El entorno virtual ya existe.")
        return venv

    log(f"Creando venv en {venv}...")
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    return venv


def install_dependencies(venv: Path, ext_dir: Path):
    python_exe = get_python(venv)

    log("Actualizando pip...")
    subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"], check=True)

    log("Instalando dependencias base de Python...")
    pip(venv, "install", *PACKAGES)

    log("Vinculando la extensión al entorno Python (vía .pth)...")
    try:
        is_win = platform.system() == "Windows"
        site_packages = venv / ("Lib/site-packages" if is_win else f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
        site_packages.mkdir(parents=True, exist_ok=True)
        pth_file = site_packages / f"{EXTENSION_NAME}.pth"
        pth_file.write_text(str(ext_dir.resolve()), encoding="utf-8")
        log(f"Enlace .pth creado en: {pth_file.name}")
    except Exception as e:
        log(f"Aviso al crear enlace .pth: {e}")


def download_hy3dgen_source(generate_dir: Path):
    """Descarga y extrae el código fuente de hy3dgen dentro de generate/_hy3dgen."""
    target_hy3dgen = generate_dir / "_hy3dgen"
    if (target_hy3dgen / "hy3dgen").exists():
        log("El módulo _hy3dgen ya existe dentro de la carpeta 'generate'.")
        return

    log("Descargando código fuente de hy3dgen desde GitHub...")
    target_hy3dgen.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(HUNYUAN_GITHUB_ZIP, timeout=180) as resp:
        data = resp.read()

    log("Extrayendo la estructura de hy3dgen dentro de generate/_hy3dgen...")
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

    log(f"hy3dgen configurado exitosamente en: {target_hy3dgen}")


def replace_custom_files(ext_dir: Path, generate_dir: Path):
    """Copia y reemplaza los archivos personalizados en generate/_hy3dgen/hy3dgen/shapegen."""
    target_dir = generate_dir / "_hy3dgen" / "hy3dgen" / "shapegen"
    
    if not target_dir.exists():
        log(f"Error: La ruta destino {target_dir} no existe.")
        return

    log("Reemplazando archivos personalizados de la extensión en hy3dgen/shapegen...")
    for filename in FILES_TO_REPLACE:
        src_file = ext_dir / filename
        dst_file = target_dir / filename

        if src_file.exists():
            shutil.copy2(src_file, dst_file)
            log(f"Reemplazado: {filename} -> {dst_file}")
        else:
            log(f"Aviso: No se encontró el archivo de origen {src_file}")


def install_models(venv: Path, ext_dir: Path, models_root: Path):
    python_exe = get_python(venv)
    model_path = models_root / EXTENSION_NAME
    model_path.mkdir(parents=True, exist_ok=True)

    # 1. Creamos la carpeta 'generate'
    generate_dir = model_path / "generate"
    generate_dir.mkdir(parents=True, exist_ok=True)

    # 2. Descargar e instalar código fuente de _hy3dgen DENTRO de generate/
    download_hy3dgen_source(generate_dir)

    # 3. Reemplazar los archivos personalizados en generate/_hy3dgen/hy3dgen/shapegen
    replace_custom_files(ext_dir, generate_dir)

    # 4. Definimos el subfolder hunyuan3d-dit-v2-0 DENTRO de generate/
    dit_dir = generate_dir / "hunyuan3d-dit-v2-0"
    dit_dir.mkdir(parents=True, exist_ok=True)

    # 5. Descargar la estructura base desde HuggingFace directamente en generate/
    log("Descargando estructura base del modelo Hunyuan3D...")
    dl_script = f"""
from huggingface_hub import snapshot_download

target_path = r'{generate_dir.resolve()}'

snapshot_download(
    repo_id='{HUNYUAN_HF_REPO}',
    local_dir=target_path,
    allow_patterns=[
        '*.json', '*.yaml', '*.md', 'LICENSE*', 'NOTICE*', '.gitattributes',
        'assets/*',
        'hunyuan3d-vae-v2-0/*',
        'hunyuan3d-dit-v2-0/*.json', 'hunyuan3d-dit-v2-0/*.yaml',
        'hunyuan3d-delight-v2-0/*.json', 'hunyuan3d-delight-v2-0/*.yaml',
        'hunyuan3d-paint-v2-0/*.json', 'hunyuan3d-paint-v2-0/*.yaml'
    ],
    ignore_patterns=['*.safetensors', '*.bin', '*.pt', '*.pth']
)
"""
    subprocess.run([str(python_exe), "-c", dl_script], check=True)

    # 6. Descargar el modelo GGUF cuantizado directamente dentro de generate/hunyuan3d-dit-v2-0
    gguf_file = dit_dir / GGUF_FILE

    if not gguf_file.exists():
        log("Descargando el modelo optimizado GGUF...")
        gguf_script = f"""
from huggingface_hub import hf_hub_download
import shutil

file = hf_hub_download(
    repo_id='{GGUF_REPO}',
    filename='{GGUF_FILE}'
)
shutil.copy2(file, r'{gguf_file.resolve()}')
"""
        subprocess.run([str(python_exe), "-c", gguf_script], check=True)
        log("Modelo GGUF descargado con éxito.")
    else:
        log("El modelo GGUF ya se encuentra presente.")

    # 7. Limpieza de carpetas no requeridas
    log("Limpiando carpetas no requeridas...")
    clean_script = f"""
import os, shutil
from pathlib import Path

base_path = Path(r'{model_path.resolve()}')
blacklist = set({BLACKLIST_DIRS})

for root, dirs, files in os.walk(base_path, topdown=False):
    root_path = Path(root)
    for dir_name in dirs:
        dir_path = root_path / dir_name
        if dir_name in blacklist:
            shutil.rmtree(dir_path, ignore_errors=True)
        elif dir_path.exists() and not any(dir_path.iterdir()):
            dir_path.rmdir()
"""
    subprocess.run([str(python_exe), "-c", clean_script], check=True)


def setup():
    log("Iniciando la configuración de Hunyuan3D 2 (GGUF)")

    paths = get_modly_paths()
    ext_dir = paths["extensions"] / EXTENSION_NAME

    clone_repository(ext_dir)
    venv = create_venv(ext_dir)
    install_dependencies(venv, ext_dir)
    install_models(venv, ext_dir, paths["models"])

    log("¡Instalación finalizada con éxito!")


if __name__ == "__main__":
    try:
        setup()
    except Exception as error:
        log("FALLÓ LA INSTALACIÓN")
        print(error)
    finally:
        input("\nPresiona ENTER para cerrar...")
