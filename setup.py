"""
Hunyuan3D 2.1 Extension Setup for Modly
========================================
Community-optimised setup script.  Handles:
  - Virtual-environment creation
  - PyTorch installation (CUDA / ROCm / CPU auto-detected)
  - Python dependency installation
  - Optional native extensions (diso on Linux, skipped on Windows)
  - Extension path linking so Modly can import local modules

NOTES:
  - Model downloads are handled BY MODLY via the manifest.json nodes.
    This script does NOT download weights.
  - The extension code (pipelines.py, volume_decoders.py, etc.) lives
    next to this file and is imported directly via a .pth link.
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
    "psutil",          # For memory reporting in generator
]

# Platform shortcuts
_IS_WINDOWS = platform.system() == "Windows"
_IS_LINUX   = platform.system() == "Linux"
_IS_MACOS   = platform.system() == "Darwin"


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

    torch_flavor values from Modly:
      "cuda"  -> NVIDIA CUDA (default)
      "rocm"  -> AMD ROCm (Linux only)
      "cpu"   -> CPU-only fallback
    """
    log(f"Installing PyTorch (flavor={torch_flavor}) ...")

    if torch_flavor == "rocm":
        if _IS_WINDOWS:
            log("WARNING: ROCm is not supported on Windows. Falling back to CPU.")
            torch_flavor = "cpu"
        else:
            # ROCm 7.2 is the current stable for consumer AMD GPUs on Linux
            _pip(venv, "install", "torch", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/rocm7.2")
            log("PyTorch + ROCm 7.2 installed.")
            return

    if torch_flavor == "cpu":
        _pip(venv, "install", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cpu")
        log("PyTorch (CPU-only) installed.")
        return

    # Default: CUDA — let pip resolve the latest CUDA wheel
    _pip(venv, "install", "torch", "torchvision")
    log("PyTorch (CUDA) installed.")


def _install_core_deps(venv: Path) -> None:
    """Install the pure-Python dependencies required by the pipeline."""
    log("Installing core Python dependencies ...")
    _pip(venv, "install", "--upgrade", "pip")
    _pip(venv, "install", *PACKAGES_CORE)
    log("Core dependencies installed.")


def _install_optional_native(venv: Path) -> None:
    """
    Install optional native extensions.

    diso  -> Fast Dual Marching Cubes.  Requires a C++ compiler.
           Skipped on Windows (MSVC not guaranteed in PATH).
           Attempted on Linux where GCC is usually present.
    """
    if _IS_WINDOWS:
        log("Skipping 'diso' on Windows (requires MSVC, not guaranteed).")
        log("The pipeline will use standard marching cubes (mc_algo='mc').")
        log("If you want DMC later, install Visual Studio Build Tools")
        log("and run: pip install diso --no-build-isolation")
        return

    log("Attempting to install 'diso' for faster mesh extraction ...")
    try:
        _pip(venv, "install", "diso")
        log("diso installed! You can use mc_algo='dmc' for faster extraction.")
    except Exception as e:
        log(f"diso installation failed ({e}). Falling back to mc_algo='mc'.")


def _link_extension_to_venv(venv: Path) -> None:
    """
    Create a .pth file so Python inside the venv can import the
    extension modules (pipelines.py, volume_decoders.py, etc.) directly.
    """
    log("Linking extension directory to Python environment ...")

    # Find site-packages inside the venv
    if _IS_WINDOWS:
        site_packages = venv / "Lib" / "site-packages"
    else:
        # lib/python3.X/site-packages
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
    """
    Main entry point called by Modly during extension installation.

    Parameters:
      python_exe   : Path to the system Python that will create the venv.
      ext_dir      : Extension directory (where this setup.py lives).
      torch_flavor : "cuda" | "rocm" | "cpu" — detected by Modly.
      accelerator  : Additional accelerator info (unused, reserved).
      platform_name: OS platform string (unused, reserved).
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

    # 5. Link extension code into the venv so imports work
    _link_extension_to_venv(venv)

    # ------------------------------------------------------------------ #
    # IMPORTANT: We do NOT download model weights here.                  #
    # ------------------------------------------------------------------ #
    # Modly handles model downloads per-node via manifest.json:
    #   - "Generate Mesh" node -> downloads hunyuan3d-dit-v2-1
    #   - "Texture Mesh" node  -> downloads hunyuan3d-paintpbr-v2-1
    # The user decides which nodes to download from the Models page.
    # ------------------------------------------------------------------ #

    log("=== Setup complete ===")
    log("Hunyuan3D 2.1 extension is ready.")
    log("Download the 'Generate Mesh' model from the Modly Models page to start.")


if __name__ == "__main__":
    # Modly calls setup.py with a JSON string as the first argument
    if len(sys.argv) >= 2:
        try:
            args = json.loads(sys.argv[1])
        except json.JSONDecodeError:
            # Fallback for older Modly versions that pass positional args
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