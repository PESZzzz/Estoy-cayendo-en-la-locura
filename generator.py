"""
Hunyuan3D 2.1 Generator for Modly — Community-optimised for AMD / low-VRAM

Model source : tencent/Hunyuan3D-2.1 (HuggingFace)
Code source  : https://github.com/PESZzzz/Estoy-cayendo-en-la-locura

This extension uses Modly's per-node download system:
  - "Generate Mesh" node : downloads only the shape model (hunyuan3d-dit-v2-1)
  - "Texture Mesh" node  : downloads only the texture model (hunyuan3d-paintpbr-v2-1)

The user can download shape first, then texture later — never forced to
pull everything at once.

Community optimisations applied:
  - Thread limiting for consumer CPUs (prevents system freeze)
  - Safe torch.compile() disabling on Windows/CPU
  - Manual sequential CPU offload for GPUs with < 8 GB VRAM
  - CPU-optimised mesh extraction with full core utilisation
  - Pre-allocated tensors in volume decoders (no 200% memory spikes)
  - Aggressive VRAM cleanup between pipeline stages
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

# =============================================================================
# Extension layout
# =============================================================================
# Modly clones this repo into its extensions folder.  The code files
# (pipelines.py, volume_decoders.py, surface_extractors.py) live next to
# this generator.py.  We add the extension directory to sys.path so
# Python can find our local modules without needing a pip install.
# =============================================================================
_EXTENSION_DIR = Path(__file__).parent
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
    """
    Try to install onnxruntime-directml using the embedded Python pip.
    Returns True if installation succeeded.
    """
    _log("Attempting to install onnxruntime-directml...")
    python_exe = sys.executable
    _log(f"Python executable: {python_exe}")

    packages = [
        "onnxruntime-directml",
        "onnxruntime",
    ]

    for pkg in packages:
        _log(f"Trying to install: {pkg}")
        try:
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", pkg, "--quiet"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                _log(f"Successfully installed {pkg}")
                return True
            else:
                _log(f"pip install failed for {pkg}: {result.stderr[:200]}")
        except Exception as e:
            _log(f"Exception during pip install of {pkg}: {e}")

    _log("All onnxruntime installation attempts failed.")
    return False


def _remove_background_pil(img: Image.Image) -> Image.Image:
    """
    Simple background removal using PIL threshold.
    Detects white/near-white pixels and makes them transparent.
    """
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
    _log("PIL fallback: background removal complete.")
    return img


class Hunyuan3D21Generator(BaseGenerator):
    MODEL_ID = "hunyuan3d-2-1"
    DISPLAY_NAME = "Hunyuan3D 2.1"
    VRAM_GB = 6

    # ------------------------------------------------------------------ #
    # Shared weights directory                                           #
    # ------------------------------------------------------------------ #
    # Modly creates one sub-directory per node:
    #   hunyuan3d-2-1/generate/   ← shape model goes here
    #   hunyuan3d-2-1/texture/    ← texture model goes here
    # We keep a reference to the parent so both nodes can find their
    # respective subfolders inside the HF repo.
    # ------------------------------------------------------------------ #

    @property
    def _weights_dir(self) -> Path:
        """
        model_dir = /hunyuan3d-2-1/{generate|texture}
        weights   = /hunyuan3d-2-1/ (one level up)
        """
        if self.model_dir.name in ("generate", "texture"):
            return self.model_dir.parent
        return self.model_dir

    # ------------------------------------------------------------------ #
    # Download                                                           #
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        """
        Modly calls this per node.  Each node has its own model_dir,
        so we check the file that corresponds to the CURRENT node.
        """
        if self.model_dir.name == "texture":
            check_file = self.model_dir / _PAINT_SUBFOLDER / "model_index.json"
        else:
            # Default to shape node
            check_file = self.model_dir / _DIT_SUBFOLDER / "config.yaml"

        result = check_file.exists()
        _log(f"is_downloaded() [{self.model_dir.name}] -> {result} (checking {check_file})")
        return result

    # ------------------------------------------------------------------ #
    # Load / Unload                                                      #
    # ------------------------------------------------------------------ #

    def load(self) -> None:
        """Load pipeline into memory from local code + HF weights."""
        _log("=== load() START ===")
        if self._model is not None:
            _log("Model already loaded, skipping load().")
            return

        import torch

        _log(f"model_dir={self.model_dir}")
        _log(f"_weights_dir={self._weights_dir}")
        _log(f"Memory BEFORE load: {_mem_info()}")

        # Ensure our local modules are importable
        if str(_EXTENSION_DIR) not in sys.path:
            sys.path.insert(0, str(_EXTENSION_DIR))

        # Import the community-optimised pipeline from the local repo
        try:
            _log("Importing local pipelines module...")
            import pipelines as local_pipelines
            _log("Import successful.")
        except Exception as e:
            _log(f"FAILED to import local pipelines: {e}")
            raise RuntimeError(f"Failed to import local pipelines from {_EXTENSION_DIR}: {e}") from e

        PipelineClass = getattr(local_pipelines, "Hunyuan3DDiTFlowMatchingPipeline", None)
        if PipelineClass is None:
            _log("ERROR: Hunyuan3DDiTFlowMatchingPipeline not found in local pipelines.")
            raise RuntimeError("Hunyuan3DDiTFlowMatchingPipeline not found.")

        _log(f"Pipeline class found: {PipelineClass.__name__}")

        device = "cuda" if torch.cuda.is_available() else ("mps" if sys.platform == "darwin" and torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32

        _log(f"Selected device: {device} | dtype: {dtype}")

        # The shape weights live inside the "generate" node directory
        shape_weights_dir = self._weights_dir / "generate"
        _log(f"Loading pipeline from: {shape_weights_dir}")

        try:
            self._model = PipelineClass.from_pretrained(
                str(shape_weights_dir),
                subfolder=_DIT_SUBFOLDER,
                use_safetensors=False,   # 2.1 ships .ckpt by default
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
                _log("CUDA cache cleared.")
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
                _log("MPS cache cleared.")
            else:
                _log("No GPU cache to clear.")
        except ImportError:
            _log("torch not available for cache clearing.")
        _log(f"Memory AFTER unload: {_mem_info()}")
        _log("=== unload() END ===")

    # ------------------------------------------------------------------ #
    # Entry point — Modly calls generate() for every node                #
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        """
        Dispatch to the correct handler based on which node is running.
        Modly sets self.model_dir to the node's directory, so we can
        tell whether we are in 'generate' or 'texture' mode.
        """
        node_name = self.model_dir.name
        _log(f"=== generate() START [node={node_name}] ===")

        if node_name == "texture":
            return self._run_texture(image_bytes, params, progress_cb, cancel_event)
        else:
            return self._run_generate(image_bytes, params, progress_cb, cancel_event)

    # ------------------------------------------------------------------ #
    # Generate Mesh node                                                 #
    # ------------------------------------------------------------------ #

    def _run_generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        _log(f"Params received: {params}")
        _log(f"image_bytes size: {len(image_bytes)} bytes")
        _log(f"Memory at generate start: {_mem_info()}")

        import torch

        num_steps      = int(params.get("num_inference_steps", 30))
        vert_count     = int(params.get("vertex_count", 0))
        octree_res     = int(params.get("octree_resolution", 256))
        guidance_scale = float(params.get("guidance_scale", 5.0))
        seed           = int(params.get("seed", -1))

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
            _log(f"Random seed generated: {seed}")
        else:
            _log(f"Fixed seed: {seed}")

        try:
            self._report(progress_cb, 5, "Removing background from image...")
            _log("Starting _preprocess()...")
            image = self._preprocess(image_bytes)
            _log(f"Preprocess done. Image size: {image.size}, mode: {image.mode}")
        except Exception as e:
            _log(f"FAILED in _preprocess(): {e}")
            traceback.print_exc()
            raise

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
            _log("Smooth progress thread started.")

        try:
            _log("Starting 3D model inference (DiT)...")
            _log(f"Config: steps={num_steps}, octree={octree_res}, guidance={guidance_scale}, seed={seed}")
            _log(f"Memory BEFORE inference: {_mem_info()}")

            with torch.inference_mode():
                generator = torch.Generator().manual_seed(seed)
                # Lower num_chunks on CPU to reduce Python call overhead
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
            _log(f"Inference completed. Mesh type: {type(mesh).__name__}")
            _log(f"Memory AFTER inference: {_mem_info()}")
        except Exception as e:
            _log(f"FAILED during 3D model inference: {e}")
            traceback.print_exc()
            raise
        finally:
            stop_evt.set()
            _log("Smooth progress thread stopped.")

        self._check_cancelled(cancel_event)

        if vert_count > 0 and hasattr(mesh, "vertices") and len(mesh.vertices) > vert_count:
            self._report(progress_cb, 90, "Optimizing vertex count...")
            _log(f"Decimating mesh: {len(mesh.vertices)} vertices -> target ~{vert_count}")
            try:
                mesh = self._decimate(mesh, vert_count)
                _log(f"Decimation done. Final vertices: {len(mesh.vertices)}")
            except Exception as e:
                _log(f"FAILED in decimation: {e}")
                traceback.print_exc()
                raise

        try:
            self.outputs_dir.mkdir(parents=True, exist_ok=True)
            name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
            path = self.outputs_dir / name
            _log(f"Exporting mesh to: {path}")
            mesh.export(str(path))
            _log(f"Export successful. File exists: {path.exists()}")
        except Exception as e:
            _log(f"FAILED to export mesh: {e}")
            traceback.print_exc()
            raise

        self._report(progress_cb, 100, "3D Generation Complete!")
        _log(f"=== _run_generate() END === | File: {path}")
        return path

    # ------------------------------------------------------------------ #
    # Texture Mesh node (stub — not implemented yet)                     #
    # ------------------------------------------------------------------ #
    # The texture pipeline of Hunyuan3D 2.1 uses a completely different
    # architecture (UNet2p5DConditionModel with PBR output).  It requires
    # compiling custom rasterizers and a 6.89 GB diffusers pipeline.
    # This stub is here so the node appears in Modly and can be downloaded
    # separately; the actual implementation will come in a future update.
    # ------------------------------------------------------------------ #

    def _run_texture(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        _log("=== _run_texture() START ===")
        _log("WARNING: Texture node is not yet implemented for Hunyuan3D 2.1")
        _log("The PBR texture pipeline requires custom native extensions")
        _log("that are still being adapted for AMD / low-VRAM systems.")

        # For now, raise a clear error so the user knows what's up
        raise NotImplementedError(
            "Hunyuan3D 2.1 texture generation is not yet available in this community build.\n"
            "Please use the 'Generate Mesh' node for shape-only output.\n"
            "Texture support will be added in a future update."
        )

    # ------------------------------------------------------------------ #
    # Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        _log("=== _preprocess() START ===")

        try:
            img = Image.open(io.BytesIO(image_bytes))
            _log(f"Image opened: {img.size}, mode={img.mode}")
        except Exception as e:
            _log(f"FAILED to open image: {e}")
            raise

        # --- Try rembg with BaseException catch (includes SystemExit) ---
        if "rembg" in sys.modules:
            mod_file = getattr(sys.modules["rembg"], "__file__", "") or ""
            if "_hy3dgen" in mod_file:
                _log("Conflicting rembg detected in sys.modules, removing...")
                del sys.modules["rembg"]

        saved_path = list(sys.path)
        sys.path = [p for p in sys.path if not p.rstrip("\\/").endswith("_hy3dgen")]

        rembg_success = False

        try:
            _log("Importing rembg...")
            import rembg
            _log("rembg imported successfully.")
            try:
                _log("Trying rembg.remove() without specific session...")
                result = rembg.remove(img).convert("RGBA")
                _log("rembg.remove() successful.")
                rembg_success = True
                return result
            except BaseException as e:
                _log(f"rembg.remove() failed ({type(e).__name__}: {e}), trying CPU session...")
                try:
                    session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
                    result = rembg.remove(img, session=session).convert("RGBA")
                    _log("rembg.remove() with CPU session successful.")
                    rembg_success = True
                    return result
                except BaseException as e2:
                    _log(f"rembg.remove() with CPU session also failed: {type(e2).__name__}: {e2}")
        except BaseException as e:
            _log(f"rembg not available or failed to import: {type(e).__name__}: {e}")
        finally:
            sys.path = saved_path
            _log("sys.path restored.")

        # --- If rembg failed, try to install onnxruntime and retry once ---
        if not rembg_success:
            _log("rembg failed. Attempting to install onnxruntime automatically...")
            if _install_onnxruntime():
                _log("onnxruntime installed. Retrying rembg...")
                try:
                    if "rembg" in sys.modules:
                        del sys.modules["rembg"]
                    if "onnxruntime" in sys.modules:
                        del sys.modules["onnxruntime"]

                    import rembg
                    result = rembg.remove(img).convert("RGBA")
                    _log("rembg.remove() successful after installation.")
                    return result
                except BaseException as e:
                    _log(f"rembg still failed after installation: {type(e).__name__}: {e}")
            else:
                _log("onnxruntime installation failed or not possible.")

        # --- Final fallback: PIL background removal ---
        _log("USING FALLBACK: PIL background removal...")
        return _remove_background_pil(img)

    def _decimate(self, mesh, target_vertices: int):
        _log(f"=== _decimate() START === | target_vertices={target_vertices}")
        target_faces = max(4, target_vertices * 2)
        try:
            result = mesh.simplify_quadric_decimation(target_faces)
            _log(f"Decimation successful. Faces before: {mesh.faces.shape[0]}, after: {result.faces.shape[0]}")
            return result
        except Exception as exc:
            _log(f"Decimation skipped: {exc}")
            traceback.print_exc()
            return mesh
        finally:
            _log("=== _decimate() END ===")