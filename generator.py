"""
Hunyuan3D 2 Generator for Modly -- Auto-install onnxruntime + PIL fallback
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

_PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0-turbo"
_DIT_SUBFOLDER = "hunyuan3d-dit-v2-0"


def _log(msg: str) -> None:
    """Log with timestamp and prefix for easy filtering in Modly logs."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    safe_msg = msg.encode("ascii", "replace").decode("ascii")
    print(f"[HY3D-DEBUG {ts}] {safe_msg}", flush=True)


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


def _install_onnxruntime() -> bool:
    """
    Try to install onnxruntime-directml using the embedded Python pip.
    Returns True if installation succeeded.
    """
    _log("Attempting to install onnxruntime-directml...")

    # Find the python executable in Modly's embedded environment
    python_exe = sys.executable
    _log(f"Python executable: {python_exe}")

    # Try different onnxruntime packages in order of preference for AMD/CPU
    packages = [
        "onnxruntime-directml",   # Best for AMD on Windows
        "onnxruntime",             # CPU-only fallback
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


class Hunyuan3DGenerator(BaseGenerator):
    MODEL_ID = "hunyuan3d-v2"
    DISPLAY_NAME = "Hunyuan3D 2"
    VRAM_GB = 6

    def is_downloaded(self) -> bool:
        generate_dir = self.model_dir / "generate"
        model_dir = generate_dir / _DIT_SUBFOLDER
        result = (
            model_dir.exists()
            and (model_dir / "model.fp16.safetensors").exists()
        )
        _log(f"is_downloaded() -> {result} (model_dir={model_dir})")
        return result

    def load(self) -> None:
        """Load pipeline into memory from _hy3dgen."""
        _log("=== load() START ===")
        if self._model is not None:
            _log("Model already loaded, skipping load().")
            return

        import torch

        generate_dir = self.model_dir / "generate" if (self.model_dir / "generate").exists() else self.model_dir
        ext_dir = Path(__file__).parent

        _log(f"model_dir={self.model_dir}")
        _log(f"generate_dir={generate_dir}")
        _log(f"ext_dir={ext_dir}")
        _log(f"Memory BEFORE load: {_mem_info()}")

        for root in [generate_dir, ext_dir, self.model_dir]:
            if root.exists() and str(root.resolve()) not in sys.path:
                _log(f"Inserting into sys.path: {root.resolve()}")
                sys.path.insert(0, str(root.resolve()))

        sys.path = [p for p in sys.path if not p.rstrip("\\/").endswith("_hy3dgen")]
        _log(f"sys.path filtered. Length: {len(sys.path)}")

        try:
            _log("Importing _hy3dgen.shapegen.pipelines...")
            import _hy3dgen.shapegen.pipelines as shapegen_pipelines
            _log("Import successful.")
        except Exception as e:
            _log(f"FAILED to import _hy3dgen.shapegen.pipelines from {generate_dir}: {e}")
            raise RuntimeError(f"Failed to import _hy3dgen.shapegen.pipelines from {generate_dir}: {e}") from e

        HunyuanPipelineClass = getattr(shapegen_pipelines, "Hunyuan3DDiTFlowMatchingPipeline", None) or \
                               getattr(shapegen_pipelines, "Hunyuan3DDiTPipeline", None)

        if HunyuanPipelineClass is None:
            _log("ERROR: Pipeline class not found in _hy3dgen.shapegen.pipelines.")
            raise RuntimeError("Pipeline class not found in _hy3dgen.shapegen.pipelines.")

        _log(f"Pipeline class found: {HunyuanPipelineClass.__name__}")

        device = "cuda" if torch.cuda.is_available() else ("mps" if sys.platform == "darwin" and torch.backends.mps.is_available() else "cpu")
        dtype = torch.float16 if device == "cuda" else torch.float32

        _log(f"Selected device: {device} | dtype: {dtype}")

        target_model_path = generate_dir / _DIT_SUBFOLDER if (generate_dir / _DIT_SUBFOLDER).exists() else generate_dir
        _log(f"Loading pipeline from: {target_model_path}")

        try:
            self._model = HunyuanPipelineClass.from_pretrained(
                str(generate_dir if (generate_dir / _DIT_SUBFOLDER).exists() else target_model_path),
                subfolder=_DIT_SUBFOLDER if (generate_dir / _DIT_SUBFOLDER).exists() else None,
                use_safetensors=True,
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

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        _log("=== generate() START ===")
        _log(f"Params received: {params}")
        _log(f"image_bytes size: {len(image_bytes)} bytes")
        _log(f"Memory at generate start: {_mem_info()}")

        import torch

        num_steps      = int(params.get("num_inference_steps", 30))
        vert_count     = int(params.get("vertex_count", 0))
        enable_texture = bool(params.get("enable_texture", False))
        octree_res     = int(params.get("octree_resolution", 384))
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

        shape_end = 70 if enable_texture else 85
        self._report(progress_cb, 12, "Generating 3D structure with Hunyuan3D DiT...")
        _log(f"enable_texture={enable_texture}, shape_end={shape_end}")

        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 12, shape_end, "Sampling 3D latent vectors...", stop_evt),
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
                # Higher num_chunks on CPU reduces Python call overhead
                # Marching cubes is inherently slow on CPU, fewer chunks = less overhead
                import torch as _torch
                effective_chunks = 8000 if not _torch.cuda.is_available() else 1000

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

        if enable_texture:
            try:
                self._report(progress_cb, 72, "Freeing VRAM for texture model...")
                _log("Starting texturing...")
                self.unload()
                self._check_cancelled(cancel_event)
                mesh = self._run_texture(mesh, image, progress_cb)
                self.load()
            except Exception as e:
                _log(f"FAILED in texturing: {e}")
                traceback.print_exc()
                raise
        else:
            if vert_count > 0 and hasattr(mesh, "vertices") and len(mesh.vertices) > vert_count:
                self._report(progress_cb, 88, "Optimizing vertex count...")
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
        _log(f"=== generate() END === | File: {path}")
        return path

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        _log("=== _preprocess() START ===")

        # Open image first so we always have a fallback
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
        rembg_error = None

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
                    rembg_error = e2
        except BaseException as e:
            _log(f"rembg not available or failed to import: {type(e).__name__}: {e}")
            rembg_error = e
        finally:
            sys.path = saved_path
            _log("sys.path restored.")

        # --- If rembg failed, try to install onnxruntime and retry once ---
        if not rembg_success:
            _log("rembg failed. Attempting to install onnxruntime automatically...")
            if _install_onnxruntime():
                _log("onnxruntime installed. Retrying rembg...")
                try:
                    # Force reimport of rembg after installation
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

    def _run_texture(self, mesh, image: Image.Image, progress_cb=None):
        _log("=== _run_texture() START ===")
        _log(f"Memory BEFORE texturing: {_mem_info()}")

        import torch
        from _hy3dgen.texgen import Hunyuan3DPaintPipeline
        from _hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        generate_dir = self.model_dir / "generate" if (self.model_dir / "generate").exists() else self.model_dir
        paint_dir = generate_dir / "_paint_weights"
        _log(f"paint_dir={paint_dir}")

        if not paint_dir.exists():
            _log(f"ERROR: paint_dir does not exist: {paint_dir}")
            raise RuntimeError(f"Texture weights (_paint_weights) not found at {paint_dir}.")

        try:
            _log("Loading Hunyuan3DPaintPipeline...")
            paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
                str(paint_dir), subfolder=_PAINT_SUBFOLDER
            )
            _log("Paint pipeline loaded.")
        except Exception as e:
            _log(f"FAILED to load Hunyuan3DPaintPipeline: {e}")
            traceback.print_exc()
            raise

        paint_pipeline.config.render_size  = 1024
        paint_pipeline.config.texture_size = 1024
        paint_pipeline.render = MeshRender(default_resolution=1024, texture_size=1024)
        _log("Render/pipeline config applied (1024x1024).")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            image.save(tmp.name)
            tmp.close()
            _log(f"Temp image saved to: {tmp.name}")

            self._report(progress_cb, 83, "Generating UV texture map...")
            _log("Starting texture inference...")
            _log(f"Memory BEFORE texture inference: {_mem_info()}")

            with torch.inference_mode():
                result = paint_pipeline(mesh, image=tmp.name)

            _log("Texture inference completed.")
            _log(f"Memory AFTER texture inference: {_mem_info()}")
        except Exception as e:
            _log(f"FAILED during texture inference: {e}")
            traceback.print_exc()
            raise
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
                _log(f"Temp file removed: {tmp.name}")
            del paint_pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                _log("CUDA cache cleared after texturing.")
            _log(f"Memory after cleanup: {_mem_info()}")
            _log("=== _run_texture() END ===")

        return result[0] if isinstance(result, (list, tuple)) else result

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

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "num_inference_steps",
                "label":   "Sampling Steps",
                "type":    "select",
                "default": 30,
                "options": [
                    {"value": 15, "label": "Fast (15 steps)"},
                    {"value": 30, "label": "Balanced (30 steps)"},
                    {"value": 50, "label": "High Quality (50 steps)"},
                ],
                "tooltip": "Number of diffusion iterations.",
            },
            {
                "id":      "octree_resolution",
                "label":   "Mesh Resolution (Octree)",
                "type":    "select",
                "default": 256,
                "options": [
                    {"value": 256, "label": "Low/Safe (256)"},
                    {"value": 384, "label": "Medium (384)"},
                    {"value": 512, "label": "High Quality (512)"},
                ],
                "tooltip": "Resolution for geometric reconstruction in Marching Cubes.",
            },
            {
                "id":      "guidance_scale",
                "label":   "Guidance Scale (CFG)",
                "type":    "float",
                "default": 5.0,
                "min":     1.0,
                "max":     10.0,
                "step":    0.5,
                "tooltip": "Fidelity intensity to input image.",
            },
            {
                "id":      "seed",
                "label":   "Seed",
                "type":    "int",
                "default": -1,
                "min":     -1,
                "max":     2147483647,
                "tooltip": "Seed for reproducibility (-1 for random).",
            },
        ]
