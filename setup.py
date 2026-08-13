"""
Hunyuan3D 2.1 Extension Setup for Modly
========================================
Community-optimised setup script.  Handles:
  - Virtual-environment creation
  - PyTorch installation (CUDA / ROCm / CPU auto-detected)
  - Python dependency installation
  - Download of hy3dshape/ source code from HuggingFace Space
  - Optional native extensions (diso on Linux, skipped on Windows)
  - Extension path linking so Modly can import local modules

NOTES:
  - Model weights are handled BY MODLY via manifest.json nodes.
    This script only downloads the Python source code (hy3dshape/)
    from the official HuggingFace Space, because the model repo
    (tencent/Hunyuan3D-2.1) does NOT include the hy3dshape folder.
  - hy3dpaint/ is included in the model repo, so Modly downloads it
    automatically when the user chooses the "Texture Mesh" node.
"""

import json
import os
import platform
import subprocess
import sys
import urllib.request
from pathlib import Path


# ------------------------------------------------------------------ #
# Configuration                                                      #
# ------------------------------------------------------------------ #
EXTENSION_NAME = "hunyuan3d-2-1"
EXTENSION_DIR = Path(__file__).parent.resolve()

# HuggingFace Space that contains the hy3dshape source code
HF_SPACE_REPO = "spaces/tencent/Hunyuan3D-2.1"
HY3DSHAPE_PREFIX = "hy3dshape/"

# Core Python packages required by Hunyuan3D 2.1
PACKAGES_CORE = [
    "Pillow",
    "numpy",
    "trimesh",
    "pymeshlab",
    "opencv-python-headless",
    "huggingface_hub",
    "safetensors",
    "diffusers>=0.32.0",
    "transformers>=4.46.0",
    "accelerate>=0.17.0",
    "einops",
    "scipy",
    "scikit-image",
    "rembg",
    "mcubes",
    "tqdm",
    "pyyaml",
    "psutil",
]

# Platform shortcuts
_IS_WINDOWS = platform.system() == "Windows"
_IS_LINUX   = platform.system() == "Linux"

# File extensions we download from the Space (code + configs)
CODE_EXTS = {".py", ".yaml", ".yml", ".json", ".md", ".txt", ".sh", ".cfg", ".ini"}
# File extensions we skip (weights)
WEIGHT_EXTS = {".safetensors", ".ckpt", ".bin", ".pth", ".pt", ".onnx", ".gguf"}


def log(msg: str) -> None:
    """Print a prefixed log line so the user can follow setup progress."""
    print(f"[setup] {msg}", flush=True)


def _run(cmd: list, check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a subprocess command with unified error handling."""
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        log(f"Command failed: {' '.join(cmd)}")
        log(f"stdout: {e.stdout[:500]}")
        log(f"stderr: {e.stderr[:500]}")
        raise


def _python_in_venv(venv: Path) -> Path:
    """Return the Python executable inside the created venv."""
    return venv / ("Scripts/python.exe" if _IS_WINDOWS else "bin/python")


def _pip(venv: Path, *args: str) -> None:
    """Run pip inside the venv.  Raises on failure."""
    py = _python_in_venv(venv)
    _run([str(py), "-m", "pip", *args])


def _create_venv(python_exe: str, ext_dir: Path) -> Path:
    """Create the extension virtual environment."""
    venv = ext_dir / "venv"
    if venv.exists():
        log(f"Re-using existing venv at {venv}")
        return venv

    log(f"Creating virtual environment in {venv} ...")
    _run([python_exe, "-m", "venv", str(venv)])
    log("venv created.")
    return venv


def _install_torch(venv: Path, torch_flavor: str) -> None:
    """
    Install PyTorch with the correct index URL for the detected accelerator.
    """
    log(f"Installing PyTorch (flavor={torch_flavor}) ...")

    if torch_flavor == "rocm":
        if _IS_WINDOWS:
            log("WARNING: ROCm is not supported on Windows. Falling back to CPU.")
            torch_flavor = "cpu"
        else:
            _pip(venv, "install", "torch", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/rocm7.2")
            log("PyTorch + ROCm 7.2 installed.")
            return

    if torch_flavor == "cpu":
        _pip(venv, "install", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cpu")
        log("PyTorch (CPU-only) installed.")
        return

    # Default: CUDA
    _pip(venv, "install", "torch", "torchvision")
    log("PyTorch (CUDA) installed.")


def _install_core_deps(venv: Path) -> None:
    """Install the pure-Python dependencies required by the pipeline."""
    log("Installing core Python dependencies ...")
    _pip(venv, "install", "--upgrade", "pip")
    _pip(venv, "install", *PACKAGES_CORE)
    log("Core dependencies installed.")


def _install_optional_native(venv: Path) -> None:
    """Install optional native extensions (diso)."""
    if _IS_WINDOWS:
        log("Skipping 'diso' on Windows (requires MSVC, not guaranteed).")
        log("The pipeline will use standard marching cubes (mc_algo='mc').")
        return

    log("Attempting to install 'diso' for faster mesh extraction ...")
    try:
        _pip(venv, "install", "diso")
        log("diso installed! You can use mc_algo='dmc' for faster extraction.")
    except Exception as e:
        log(f"diso installation failed ({e}). Falling back to mc_algo='mc'.")


def _download_hy3dshape_from_space(ext_dir: Path) -> None:
    """
    Download the hy3dshape/ source folder from the official HuggingFace Space.

    The model repo (tencent/Hunyuan3D-2.1) does NOT include hy3dshape/.
    It only lives inside the Space (spaces/tencent/Hunyuan3D-2.1).
    We fetch only code/config files, skipping any model weights.
    """
    log("Downloading hy3dshape/ source code from HuggingFace Space ...")
    log(f"Space: {HF_SPACE_REPO}")

    api_url = f"https://huggingface.co/api/spaces/tencent/Hunyuan3D-2.1/tree/main?recursive=true"
    base_url = f"https://huggingface.co/{HF_SPACE_REPO}/resolve/main/"

    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        log(f"WARNING: Could not list files from Space API ({e}).")
        log("The extension may not work without hy3dshape/.")
        log("Please download it manually from:")
        log("  https://huggingface.co/spaces/tencent/Hunyuan3D-2.1/tree/main/hy3dshape")
        return

    downloaded = 0
    skipped = 0

    for entry in data:
        path = entry.get("path", "")
        etype = entry.get("type", "")
        size = entry.get("size", 0)

        if etype == "directory":
            continue

        # Only files inside hy3dshape/
        if not path.startswith(HY3DSHAPE_PREFIX):
            skipped += 1
            continue

        ext = os.path.splitext(path)[1].lower()

        if ext in WEIGHT_EXTS:
            skipped += 1
            continue

        # Target path inside the extension directory
        relative = path[len(HY3DSHAPE_PREFIX):]  # strip "hy3dshape/"
        local_path = ext_dir / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)

        url = base_url + path
        try:
            urllib.request.urlretrieve(url, str(local_path))
            downloaded += 1
            log(f"  + {relative}")
        except Exception as e:
            log(f"  ! FAILED {relative}: {e}")

    log(f"hy3dshape/ download complete: {downloaded} files downloaded, {skipped} skipped.")


def _link_extension_to_venv(venv: Path, ext_dir: Path) -> None:
    """
    Create a .pth file so Python inside the venv can import the
    extension modules (pipelines.py, volume_decoders.py, hy3dshape/, etc.)
    directly without a pip install.
    """
    log("Linking extension directory to Python environment ...")

    if _IS_WINDOWS:
        site_packages = venv / "Lib" / "site-packages"
    else:
        lib_dir = venv / "lib"
        candidates = sorted(lib_dir.glob("python3.*/site-packages")) if lib_dir.exists() else []
        if not candidates:
            raise RuntimeError(f"Could not find site-packages inside {venv}")
        site_packages = candidates[-1]

    site_packages.mkdir(parents=True, exist_ok=True)
    pth_file = site_packages / f"{EXTENSION_NAME}.pth"
    pth_file.write_text(str(ext_dir), encoding="utf-8")
    log(f".pth link created: {pth_file}")


def setup(
    python_exe: str,
    ext_dir: Path,
    gpu_sm: int = 0,
    cuda_version: int = 0,
    torch_flavor: str = "cuda",
    accelerator: str = "",
    platform_name: str = "",
) -> None:
    """
    Main entry point called by Modly during extension installation.
    """
    log(f"=== {EXTENSION_NAME} setup starting ===")
    log(f"Extension directory: {ext_dir}")
    log(f"Platform: {platform.system()} | torch_flavor: {torch_flavor}")

    # 1. Create virtual environment
    venv = _create_venv(python_exe, ext_dir)

    # 2. Install PyTorch with the correct backend
    _install_torch(venv, torch_flavor)

    # 3. Install core Python dependencies
    _install_core_deps(venv)

    # 4. Optional native extensions
    _install_optional_native(venv)

    # 5. Download hy3dshape/ source code from HF Space
    #    (The model repo does not include it — it only lives in the Space)
    _download_hy3dshape_from_space(ext_dir)

    # 6. Link extension code into the venv so imports work
    _link_extension_to_venv(venv, ext_dir)

    # ------------------------------------------------------------------ #
    # IMPORTANT: Model weights are NOT downloaded here.                  #
    # Modly handles them per-node via manifest.json:                     #
    #   - "Generate Mesh" node -> downloads hunyuan3d-dit-v2-1           #
    #   - "Texture Mesh" node  -> downloads hunyuan3d-paintpbr-v2-1      #
    # ------------------------------------------------------------------ #

    log("=== Setup complete ===")
    log("Hunyuan3D 2.1 extension is ready.")
    log("Download the 'Generate Mesh' model from the Modly Models page to start.")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        try:
            args = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            args = {
                "python_exe": sys.argv[1],
                "ext_dir": sys.argv[2] if len(sys.argv) > 2 else str(Path(__file__).parent),
                "gpu_sm": int(sys.argv[3]) if len(sys.argv) > 3 else 0,
                "cuda_version": int(sys.argv[4]) if len(sys.argv) > 4 else 0,
                "torch_flavor": sys.argv[5] if len(sys.argv) > 5 else "cuda",
            }

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
        print("Usage: python setup.py '<json_args>'")
        sys.exit(1)
