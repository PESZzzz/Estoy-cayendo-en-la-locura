"""
Hunyuan3D 2 Generator for Modly (Optimized)
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
        model_dir = self.model_dir / _DIT_SUBFOLDER
        return (
            model_dir.exists()
            and (model_dir / "model.fp16.safetensors").exists()
        )

    def load(self) -> None:
        """Carga el pipeline a memoria usando el motor hy3dgen parcheado."""
        if self._model is not None:
            return

        # 1. Búsqueda y resolución de rutas para la librería 'hy3dgen'
        ext_dir = Path(__file__).parent
        possible_roots = [
            self.model_dir,
            self.model_dir.parent,
            ext_dir,
            ext_dir / "_hy3dgen",
            self.model_dir / "_hy3dgen",
        ]

        found = False
        for root in possible_roots:
            if not root.exists():
                continue
            
            if (root / "hy3dgen").is_dir():
                if str(root) not in sys.path:
                    sys.path.insert(0, str(root))
                found = True
                break
            elif (root / "_hy3dgen" / "hy3dgen").is_dir():
                target = root / "_hy3dgen"
                if str(target) not in sys.path:
                    sys.path.insert(0, str(target))
                found = True
                break

        import torch

        # 2. Intentar importar la clase de pipeline híbrido
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"No se encontró la librería 'hy3dgen'. "
                f"Rutas examinadas: model_dir='{self.model_dir}' y ext_dir='{ext_dir}'. "
                f"Asegúrate de que la carpeta 'hy3dgen' esté presente en la estructura del proyecto."
            ) from e

        # Configuración de Dispositivo y Precision
        if sys.platform == "darwin":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            dtype  = torch.float32
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype  = torch.float16 if device == "cuda" else torch.float32

        target_dir  = self.model_dir / _DIT_SUBFOLDER if (self.model_dir / _DIT_SUBFOLDER).exists() else self.model_dir
        config_path = target_dir / "config.yaml"

        print(f"[Hunyuan3DGGUFGenerator] Cargando Pipeline GGUF Híbrido ({device}, {dtype}) desde: {target_dir}")
        
        # 3. Instanciación desde el pipeline refactorizado
        try:
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                str(self.model_dir),
                subfolder=_DIT_SUBFOLDER,
                use_safetensors=True,
                device=device,
                dtype=dtype,
            )
        except Exception as err:
            print(f"[Hunyuan3DGGUFGenerator] ERROR CRÍTICO al inicializar pipeline: {err}")
            raise RuntimeError(f"Fallo al instanciar el modelo GGUF de Hunyuan3D: {err}") from err

        self._model = pipeline
        print(f"[Hunyuan3DGGUFGenerator] Modelo cargado correctamente en memoria.")

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
        self._report(progress_cb, 12, "Generando estructura 3D mediante GGUF DiT...")
        
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
        self._report(progress_cb, 96, "Exportando archivo GLB...")
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

        paint_dir = self.model_dir / "_paint_weights"
        if not paint_dir.exists():
            raise RuntimeError("Los pesos de texturizado (_paint_weights) no se encuentran en el directorio del modelo.")

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
            print(f"[Hunyuan3DGGUFGenerator] Decimación omitida: {exc}")
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
                    {"value": 512, "label": "Alta (512)"},
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
