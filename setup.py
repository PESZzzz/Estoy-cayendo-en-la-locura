"""
Hunyuan3D 2.1 Extension Setup for Modly
========================================
Community-optimised setup script.  Handles:
  - Virtual-environment creation
  - PyTorch installation (CUDA / ROCm / CPU auto-detected)
  - Python dependency installation (in small batches to avoid timeouts)
  - Optional native extensions (diso on Linux, skipped on Windows)
  - Extension path linking so Modly can import local modules

NOTES:
  - Model weights are handled BY MODLY via manifest.json nodes.
    This script only installs Python dependencies and links the code.
  - The hy3dshape/ source code is bundled with this extension.
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


# ------------------------------------------------------------------ #
# Configuration                                                      #
# ------------------------------------------------------------------ #
EXTENSION_NAME = "hunyuan3d-2-1"
EXTENSION_DIR = Path(__file__).parent.resolve()

# Dependencies split into batches to avoid Windows/embedded-Python timeouts.
BATCHES = [
    {"packages": ["--upgrade", "pip"], "timeout": 120, "label": "Upgrading pip"},
    {"packages": ["numpy", "scipy", "scikit-image"], "timeout": 600, "label": "Installing scientific stack"},
    {"packages": ["diffusers>=0.32.0", "transformers>=4.46.0", "accelerate>=0.17.0", "safetensors", "einops", "omegaconf", "timm"], "timeout": 600, "label": "Installing ML stack"},
    {"packages": ["trimesh", "pymeshlab", "mcubes", "opencv-python-headless", "Pillow", "rembg"], "timeout": 600, "label": "Installing 3D / image stack"},
    {"packages": ["huggingface_hub", "tqdm", "pyyaml", "psutil"], "timeout": 300, "label": "Installing utilities"},
]

_IS_WINDOWS = platform.system() == "Windows"
_IS_LINUX   = platform.system() == "Linux"


def log(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


def _run(cmd: list, check: bool = True, timeout: int = 300) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(cmd, check=check, capture_output=True, text=True, timeout=timeout)
    except subprocess.CalledProcessError as e:
        log(f"Command failed: {' '.join(cmd)}")
        log(f"stdout: {e.stdout[:500]}")
        log(f"stderr: {e.stderr[:500]}")
        raise


def _python_in_venv(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if _IS_WINDOWS else "bin/python")


def _pip(venv: Path, args: list, timeout: int = 300) -> None:
    py = _python_in_venv(venv)
    _run([str(py), "-m", "pip", "install", *args], timeout=timeout)


def _create_venv(python_exe: str, ext_dir: Path) -> Path:
    venv = ext_dir / "venv"
    if venv.exists():
        log(f"Re-using existing venv at {venv}")
        return venv
    log(f"Creating virtual environment in {venv} ...")
    _run([python_exe, "-m", "venv", str(venv)])
    log("venv created.")
    return venv


def _install_torch(venv: Path, torch_flavor: str) -> None:
    log(f"Installing PyTorch (flavor={torch_flavor}) ...")
    if torch_flavor == "rocm":
        if _IS_WINDOWS:
            log("WARNING: ROCm is not supported on Windows. Falling back to CPU.")
            torch_flavor = "cpu"
        else:
            _pip(venv, ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/rocm7.2"], timeout=600)
            log("PyTorch + ROCm 7.2 installed.")
            return
    if torch_flavor == "cpu":
        _pip(venv, ["torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"], timeout=600)
        log("PyTorch (CPU-only) installed.")
        return
    _pip(venv, ["torch", "torchvision"], timeout=600)
    log("PyTorch (CUDA) installed.")


def _install_core_deps(venv: Path) -> None:
    log("Installing core Python dependencies (in batches to avoid timeouts)...")
    for batch in BATCHES:
        label = batch["label"]
        packages = batch["packages"]
        timeout = batch["timeout"]
        log(f"  -> {label}: {', '.join(packages)}")
        try:
            _pip(venv, packages, timeout=timeout)
            log(f"     {label} OK")
        except subprocess.TimeoutExpired:
            log(f"     TIMEOUT after {timeout}s. Retrying once with doubled timeout...")
            _pip(venv, packages, timeout=timeout * 2)
            log(f"     {label} OK (retry)")
    log("All core dependencies installed.")


def _install_optional_native(venv: Path) -> None:
    if _IS_WINDOWS:
        log("Skipping 'diso' on Windows (requires MSVC, not guaranteed).")
        log("The pipeline will use standard marching cubes (mc_algo='mc').")
        return
    log("Attempting to install 'diso' for faster mesh extraction ...")
    try:
        _pip(venv, ["diso"], timeout=600)
        log("diso installed! You can use mc_algo='dmc' for faster extraction.")
    except Exception as e:
        log(f"diso installation failed ({e}). Falling back to mc_algo='mc'.")


def _link_extension_to_venv(venv: Path) -> None:
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
    pth_file.write_text(str(EXTENSION_DIR), encoding="utf-8")
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
    log(f"=== {EXTENSION_NAME} setup starting ===")
    log(f"Extension directory: {ext_dir}")
    log(f"Platform: {platform.system()} | torch_flavor: {torch_flavor}")
    venv = _create_venv(python_exe, ext_dir)
    _install_torch(venv, torch_flavor)
    _install_core_deps(venv)
    _install_optional_native(venv)
    _link_extension_to_venv(venv)
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
