def load(self) -> None:
        """Carga el pipeline a memoria con resolución extrema de rutas."""
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

        # Inyectar al sys.path todas las carpetas posibles para que Python tenga alcance total
        for root in possible_roots:
            if root.exists() and str(root.resolve()) not in sys.path:
                sys.path.insert(0, str(root.resolve()))

        # 2. Importar el módulo pipelines interceptando cualquier variación de nombre de carpeta
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
