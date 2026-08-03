"""
Hunyuan3D 2 Generator for Modly
"""
import gc
import io
import os
import random
import sys
import tempfile
import time
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled

_PAINT_SUBFOLDER = "hunyuan3d-paint-v2-0-turbo"
_DIT_SUBFOLDER = "hunyuan3d-dit-v2-0"


class Hunyuan3DGenerator(BaseGenerator):
    MODEL_ID = "hunyuan3d-v2"
    DISPLAY_NAME = "Hunyuan3D 2"
    VRAM_GB = 6

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def is_downloaded(self) -> bool:
        generate_dir = self.model_dir / "generate"
        model_dir = generate_dir / _DIT_SUBFOLDER
        return (
            model_dir.exists()
            and (model_dir / "model.fp16.safetensors").exists()
        )

    def load(self) -> None:
        """Carga el pipeline a memoria con resolución de rutas."""
        if self._model is not None:
            return

        import torch

        # 1. Definir rutas bases
        generate_dir = self.model_dir / "generate" if (self.model_dir / "generate").exists() else self.model_dir
        ext_dir = Path(__file__).parent

        possible_roots = [
            generate_dir / "_hy3dgen",
            generate_dir,
            ext_dir / "_hy3dgen",
            ext_dir,
            self.model_dir,
        ]

        # Inyectar al sys.path todas las carpetas posibles
        for root in possible_roots:
            if root.exists() and str(root.resolve()) not in sys.path:
                sys.path.insert(0, str(root.resolve()))

        # 2. Importar el módulo pipelines interceptando variaciones de paquete
        shapegen_pipelines = None
        import_errors = []
        
        try:
            import hy3dgen.shapegen.pipelines as shapegen_pipelines
        except ImportError as e:
            import_errors.append(f"Intento 1 (hy3dgen.shapegen): {e}")
            try:
                import _hy3dgen.shapegen.pipelines as shapegen_pipelines
            except ImportError as e2:
                import_errors.append(f"Intento 2 (_hy3dgen.shapegen): {e2}")
                try:
                    import shapegen.pipelines as shapegen_pipelines
                except ImportError as e3:
                    import_errors.append(f"Intento 3 (shapegen puro): {e3}")
                    
        if shapegen_pipelines is None:
            raise RuntimeError(f"Fallo total al importar pipelines.py. Errores:\n" + "\n".join(import_errors))

        # 3. Extraer la clase oficial
        HunyuanPipelineClass = None
        candidate_names = ["Hunyuan3DDiTPipeline", "Hunyuan3DDiTFlowMatchingPipeline", "Hunyuan3DPipeline"]
        for name in candidate_names:
            if hasattr(shapegen_pipelines, name):
                HunyuanPipelineClass = getattr(shapegen_pipelines, name)
                print(f"[Hunyuan3DGenerator] ¡Clase detectada con éxito!: {name}")
                break

        if HunyuanPipelineClass is None:
            raise RuntimeError(f"El módulo se importó correctamente, pero no contiene las clases de Pipeline esperadas.")

        # 4. Configuración de Dispositivo y Precisión
        if sys.platform == "darwin":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype  = torch.float32
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype  = torch.float16 if device == "cuda" else torch.float32

        target_model_path = generate_dir / _DIT_SUBFOLDER if (generate_dir / _DIT_SUBFOLDER).exists() else generate_dir
        print(f"[Hunyuan3DGenerator] Cargando Pipeline ({device}, {dtype}) desde: {target_model_path}")

        # 5. Instanciación del Pipeline
        try:
            if (generate_dir / _DIT_SUBFOLDER).exists():
                pipeline = HunyuanPipelineClass.from_pretrained(
                    str(generate_dir),
                    subfolder=_DIT_SUBFOLDER,
                    use_safetensors=True,
                    device=device,
                    dtype=dtype,
                )
            else:
                pipeline = HunyuanPipelineClass.from_pretrained(
                    str(target_model_path),
                    use_safetensors=True,
                    device=device,
                    dtype=dtype,
                )
        except Exception as err:
            print(f"[Hunyuan3DGenerator] ERROR CRÍTICO al inicializar pipeline: {err}")
            raise RuntimeError(f"Fallo al instanciar el modelo de Hunyuan3D desde {generate_dir}: {err}") from err

        self._model = pipeline
        print(f"[Hunyuan3DGenerator] Modelo cargado correctamente en memoria.")

    def unload(self) -> None:
        """Libera la VRAM/RAM asignada al modelo."""
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

    # ------------------------------------------------------------------ #
    # Inference
    # ------------------------------------------------------------------ #

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        import torch

        num_steps      = int(params.get("num_inference_steps", 30))
        vert_count     = int(params.get("vertex_count", 0))
        enable_texture = bool(params.get("enable_texture", False))
        octree_res     = int(params.get("octree_resolution", 384))
        guidance_scale = float(params.get("guidance_scale", 5.0))
        seed           = int(params.get("seed", -1))

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # 1. Preprocesamiento de Imagen (Remoción de Fondo)
        self._report(progress_cb, 5, "Removiendo fondo de la imagen...")
        image = self._preprocess(image_bytes)
        self._check_cancelled(cancel_event)

        shape_end = 70 if enable_texture else 85
        self._report(progress_cb, 12, "Generando estructura 3D mediante Hunyuan3D DiT...")
        
        stop_evt = threading.Event()
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 12, shape_end, "Muestreando vectores latentes 3D...", stop_evt),
                daemon=True,
            )
            t.start()

        # 2. Generación de Geometría Base (Trimesh Output)
        try:
            with torch.inference_mode():
                generator = torch.Generator().manual_seed(seed)
                outputs = self._model(
                    image=image,
                    num_inference_steps=num_steps,
                    octree_resolution=octree_res,
                    guidance_scale=guidance_scale,
                    num_chunks=4000,
                    generator=generator,
                    output_type="trimesh",
                )
            mesh = outputs[0] if isinstance(outputs, list) else outputs
        finally:
            stop_evt.set()

        self._check_cancelled(cancel_event)

        # 3. Texturizado (Opcional) o Decimación de Malla
        if enable_texture:
            self._report(progress_cb, 72, "Liberando VRAM para el modelo de texturizado...")
            self.unload()

            self._check_cancelled(cancel_event)
            mesh = self._run_texture(mesh, image, progress_cb)
            self.load()  # Re-restaura el modelo base
        else:
            if vert_count > 0 and hasattr(mesh, "vertices") and len(mesh.vertices) > vert_count:
                self._report(progress_cb, 88, "Optimizando recuento de vértices...")
                mesh = self._decimate(mesh, vert_count)

        # 4. Exportación a Formato GLB
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.glb"
        path = self.outputs_dir / name
        mesh.export(str(path))

        self._report(progress_cb, 100, "¡Generación 3D Completada!")
        return path

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _preprocess(self, image_bytes: bytes) -> Image.Image:
        import rembg
        img = Image.open(io.BytesIO(image_bytes))
        try:
            return rembg.remove(img).convert("RGBA")
        except Exception:
            session = rembg.new_session("u2net", providers=["CPUExecutionProvider"])
            return rembg.remove(img, session=session).convert("RGBA")

    def _run_texture(self, mesh, image: Image.Image, progress_cb=None):
        import torch
        from hy3dgen.texgen import Hunyuan3DPaintPipeline
        from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        generate_dir = self.model_dir / "generate" if (self.model_dir / "generate").exists() else self.model_dir
        paint_dir = generate_dir / "_paint_weights"
        if not paint_dir.exists():
            raise RuntimeError(f"Los pesos de texturizado (_paint_weights) no se encuentran en {paint_dir}.")

        self._report(progress_cb, 78, "Cargando modelo de pintura de textura...")
        paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            str(paint_dir), subfolder=_PAINT_SUBFOLDER
        )

        paint_pipeline.config.render_size  = 1024
        paint_pipeline.config.texture_size = 1024
        paint_pipeline.render = MeshRender(default_resolution=1024, texture_size=1024)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        try:
            image.save(tmp.name)
            tmp.close()

            self._report(progress_cb, 83, "Generando mapa de texturas UV...")
            with torch.inference_mode():
                result = paint_pipeline(mesh, image=tmp.name)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
            del paint_pipeline
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result[0] if isinstance(result, (list, tuple)) else result

    def _decimate(self, mesh, target_vertices: int):
        target_faces = max(4, target_vertices * 2)
        try:
            return mesh.simplify_quadric_decimation(target_faces)
        except Exception as exc:
            print(f"[Hunyuan3DGenerator] Decimación omitida: {exc}")
            return mesh

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id":      "num_inference_steps",
                "label":   "Pasos de Muestreo",
                "type":    "select",
                "default": 30,
                "options": [
                    {"value": 15, "label": "Rápido (15 pasos)"},
                    {"value": 30, "label": "Balanceado (30 pasos)"},
                    {"value": 50, "label": "Alta Calidad (50 pasos)"},
                ],
                "tooltip": "Número de iteraciones de difusión.",
            },
            {
                "id":      "octree_resolution",
                "label":   "Resolución de Malla (Octree)",
                "type":    "select",
                "default": 384,
                "options": [
                    {"value": 256, "label": "Baja (256)"},
                    {"value": 384, "label": "Media (384)"},
                    {"value": 512, "label": "Alta Calidad (512)"},
                ],
                "tooltip": "Resolución para la reconstrucción geométrica en Marching Cubes.",
            },
            {
                "id":      "guidance_scale",
                "label":   "Escala de Guía (CFG)",
                "type":    "float",
                "default": 5.0,
                "min":     1.0,
                "max":     10.0,
                "step":    0.5,
                "tooltip": "Intensidad de fidelidad respecto a la imagen de entrada.",
            },
            {
                "id":      "seed",
                "label":   "Semilla (Seed)",
                "type":    "int",
                "default": -1,
                "min":     -1,
                "max":     2147483647,
                "tooltip": "Semilla para reproducibilidad (-1 para aleatorio).",
            },
        ]
