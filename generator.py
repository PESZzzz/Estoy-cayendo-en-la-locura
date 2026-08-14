"""
Hunyuan3D 2.1 Generator for Modly -- Community-optimised for AMD / low-VRAM

Model source : tencent/Hunyuan3D-2.1 (HuggingFace)
Code source  : https://github.com/PESZzzz/Estoy-cayendo-en-la-locura

This extension uses Modly's per-node download system:
  - "Generate Mesh" node : downloads only the shape model (hunyuan3d-dit-v2-1)
  - "Texture Mesh" node  : downloads only the texture model (hunyuan3d-paintpbr-v2-1)

The hy3dshape/ source code is bundled with this extension (not downloaded
from the HF Space at install time).  Our community-optimised files live
inside hy3dshape/ replacing the upstream originals.
"""
import gc
import io
import os
import random
import subprocess
import sys
import tempfile
import time
import threading
import traceback
import uuid
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

# ------------------------------------------------------------------ #
# Extension layout                                                   #
# ------------------------------------------------------------------ #
# The extension code (including hy3dshape/) lives next to this file.
# We add the extension directory to sys.path so Python can find our
# local hy3dshape package without needing a pip install.
# ------------------------------------------------------------------ #
_EXTENSION_DIR = Path(__file__).parent.resolve()
if str(_EXTENSION_DIR) not in sys.path:
    sys.path.insert(0, str(_EXTENSION_DIR))

# Subfolders inside the HuggingFace repo for each node
_DIT_SUBFOLDER = "hunyuan3d-dit-v2-1"
_PAINT_SUBFOLDER = "hunyuan3d-paintpbr-v2-1"


def _log(msg: str) -> None:
    """Log with timestamp and prefix for easy filtering in Modly logs."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    safe_msg = msg.encode("ascii", "replace").decode("ascii")
    print(f"[HY3D21-DEBUG {ts}] {safe_msg}", flush=True)


def _mem_info() -> str:
    """Return memory info if torch is available."""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            return f"CUDA alloc={alloc:.1f}MB reserved={reserved:.1f}MB"
        elif sys.platform == "darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "MPS active (no memory metrics available)"
        else:
            import psutil
            mem = psutil.virtual_memory()
            return f"CPU RAM used={mem.used/1024**2:.1f}MB free={mem.available/1024**2:.1f}MB"
    except Exception:
        return "Could not get memory info"


def _free_vram():
    """Free residual GPU memory to prevent hangs on laptops and AMD."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _install_onnxruntime() -> bool:
    """Try to install onnxruntime-directml using the embedded Python pip."""
    _log("Attempting to install onnxruntime-directml...")
    python_exe = sys.executable
    packages = ["onnxruntime-directml", "onnxruntime"]
    for pkg in packages:
        try:
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", pkg, "--quiet"],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode == 0:
                _log(f"Successfully installed {pkg}")
                return True
        except Exception as e:
            _log(f"Exception during pip install of {pkg}: {e}")
    _log("All onnxruntime installation attempts failed.")
    return False


def _remove_background_pil(img: Image.Image) -> Image.Image:
    """Simple background removal using PIL threshold."""
    _log("PIL fallback: removing background via threshold...")
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    datas = img.getdata()
    new_data = []
    threshold = 240
    for item in datas:
        r, g, b, a = item
        if r > threshold and g > threshold and b > threshold:
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    img.putdata(new_data)
    return img


class Hunyuan3D21Generator(BaseGenerator):
    MODEL_ID = "hunyuan3d-2-1"
    DISPLAY_NAME = "Hunyuan3D 2.1"
    VRAM_GB = 6

    @property
    def _weights_dir(self) -> Path:
        """model_dir = /hunyuan3d-2-1/{generate|texture}; weights = parent."""
        if self.model_dir.name in ("generate", "texture"):
            return self.model_dir.parent
        return self.model_dir

    def is_downloaded(self) -> bool:
        if self.model_dir.name == "texture":
            check_file = self.model_dir / _PAINT_SUBFOLDER / "model_index.json"
        else:
            check_file = self.model_dir / _DIT_SUBFOLDER / "config.yaml"
        result = check_file.exists()
        _log(f"is_downloaded() [{self.model_dir.name}] -> {result} (checking {check_file})")
        return result

    def load(self) -> None:
        """Load pipeline into memory from local hy3dshape package."""
        _log("=== load() START ===")
        if self._model is not None:
            _log("Model already loaded, skipping load().")
            return

        import torch

        _log(f"model_dir={self.model_dir}")
        _log(f"_weights_dir={self._weights_dir}")
        _log(f"Memory BEFORE load: {_mem_info()}")

        # Ensure extension dir is in path so hy3dshape can be imported
        if str(_EXTENSION_DIR) not in sys.path:
            sys.path.insert(0, str(_EXTENSION_DIR))

        # Import the community-optimised pipeline from the bundled hy3dshape package
        try:
            _log("Importing hy3dshape.pipelines ...")
            import hy3dshape.pipelines as local_pipelines
            _log("Import successful.")
        except Exception as e:
            _log(f"FAILED to import hy3dshape.pipelines: {e}")
            raise RuntimeError(f"Failed to import hy3dshape.pipelines from {_EXTENSION_DIR}: {e}") from e

        PipelineClass = getattr(local_pipelines, "Hunyuan3DDiTFlowMatchingPipeline", None)
        if PipelineClass is None:
            raise RuntimeError("Hunyuan3DDiTFlowMatchingPipeline not found in hy3dshape.pipelines.")

        _log(f"Pipeline class found: {PipelineClass.__name__}")

        device = "cuda" if torch.cuda.is_available() else ("mps" if sys.platform == "darwin" and torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32
        _log(f"Selected device: {device} | dtype: {dtype}")

        shape_weights_dir = self._weights_dir / "generate"
        _log(f"Loading pipeline from: {shape_weights_dir}")

        try:
            self._model = PipelineClass.from_pretrained(
                str(shape_weights_dir),
                subfolder=_DIT_SUBFOLDER,
                use_safetensors=False,
                device=device,
                dtype=dtype,
            )
            _log(f"Model loaded successfully. Memory AFTER: {_mem_info()}")
        except Exception as e:
            _log(f"FAILED to load model: {e}")
            traceback.print_exc()
            raise

        _log("=== load() END ===")

    def unload(self) -> None:
        """Free VRAM/RAM allocated to the model."""
        _log("=== unload() START ===")
        _log(f"Memory BEFORE unload: {_mem_info()}")
        super().unload()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except ImportError:
            pass
        _log(f"Memory AFTER unload: {_mem_info()}")
        _log("=== unload() END ===")

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        node_name = self.model_dir.name
        _log(f"=== generate() START [node={node_name}] ===")
        if node_name == "texture":
            return self._run_texture(image_bytes, params, progress_cb, cancel_event)
        return self._run_generate(image_bytes, params, progress_cb, cancel_event)

    def _run_generate(self, image_bytes, params, progress_cb, cancel_event):
        _log(f"Params: {params}")
        _log(f"image_bytes: {len(image_bytes)} bytes")
        _log(f"Memory at start: {_mem_info()}")

        import torch

        num_steps      = int(params.get("num_inference_steps", 30))
        vert_count     = int(params.get("vertex_count", 0))
        octree_res     = int(params.get("octree_resolution", 256))
        guidance_scale = float(params.get("guidance_scale", 5.0))
        seed           = int(params.get("seed", -1))
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        image = self._preprocess(image_bytes)
        self._check_cancelled(cancel_event)

        self._report(progress_cb, 12, "Generating 3D structure with Hunyuan3D 2.1 DiT...")
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 12, 88, "Sampling 3D latent vectors...", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            with torch.inference_mode():
                generator = torch.Generator().manual_seed(seed)
                effective_chunks = 8000 if not torch.cuda.is_available() else 2000
                outputs = self._model(
                    image=image,
                    num_inference_steps=num_steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=effective_chunks,
                    generator=generator,
                    output_type="trimesh",
                )
            mesh = outputs[0] if isinstance(outputs, list) else outputs
            _log(f"Inference done. Mesh type: {type(mesh).__name__}")
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        if vert_count > 0 and hasattr(mesh, "vertices") and len(mesh.vertices) > vert_count:
            self._report(progress_cb, 90, "Optimizing vertex count...")
            mesh = self._decimate(mesh, vert_count)

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))

        self._report(progress_cb, 100, "3D Generation Complete!")
        return path

    def _run_texture(self, image_bytes, params, progress_cb, cancel_event):
        _log("=== _run_texture() START ===")
        raise NotImplementedError(
            "Hunyuan3D 2.1 texture generation is not yet available.\n"
            "Please use the 'Generate Mesh' node for shape-only output."
        )

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        img = Image.open(io.BytesIO(image_bytes))
        if "rembg" in sys.modules:
            mod_file = getattr(sys.modules["rembg"], "__file__", "") or ""
            if "_hy3dgen" in mod_file:
                del sys.modules["rembg"]

        saved_path = list(sys.path)
        sys.path = [p for p in sys.path if not p.rstrip("\\/").endswith("_hy3dgen")]

        try:
            import rembg
            try:
                return rembg.remove(img).convert("RGBA")
            except BaseException:
                session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
                return rembg.remove(img, session=session).convert("RGBA")
        except BaseException:
            if _install_onnxruntime():
                if "rembg" in sys.modules:
                    del sys.modules["rembg"]
                if "onnxruntime" in sys.modules:
                    del sys.modules["onnxruntime"]
                import rembg
                return rembg.remove(img).convert("RGBA")
        finally:
            sys.path = saved_path

        return _remove_background_pil(img)

    def _decimate(self, mesh, target_vertices: int):
        target_faces = max(4, target_vertices * 2)
        try:
            return mesh.simplify_quadric_decimation(target_faces)
        except Exception as exc:
            _log(f"Decimation skipped: {exc}")
            return mesh
