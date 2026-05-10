from __future__ import annotations

from atomstudio.config import RenderJobConfig

try:
    import bpy  # type: ignore
except Exception:  # pragma: no cover
    bpy = None


class RenderSetup:
    def __init__(
        self,
        *,
        engine: str = "cycles",
        device: str = "auto",
        samples: int = 64,
        resolution: tuple[int, int] = (1024, 1024),
        transparent_bg: bool = True,
        seed: int = 7,
    ) -> None:
        self.engine = str(engine)
        self.device = self._normalize_device(device)
        self.samples = int(samples)
        self.resolution = (int(resolution[0]), int(resolution[1]))
        self.transparent_bg = bool(transparent_bg)
        self.seed = int(seed)

    @classmethod
    def from_cfg(
        cls,
        cfg: RenderJobConfig,
    ) -> "RenderSetup":
        return cls(
            engine=cfg.render.engine,
            device=cfg.render.device,
            samples=cfg.render.samples,
            resolution=cfg.render.resolution,
            transparent_bg=cfg.render.transparent_bg,
            seed=cfg.render.seed,
        )

    def apply(self) -> None:
        if bpy is None:
            raise RuntimeError("bpy is not available. Run this function inside Blender.")

        scene = bpy.context.scene
        engine_name = self._engine_name(self.engine)
        scene.render.engine = engine_name
        scene.render.resolution_x = max(1, int(self.resolution[0]))
        scene.render.resolution_y = max(1, int(self.resolution[1]))
        scene.render.image_settings.file_format = "PNG"
        scene.render.film_transparent = bool(self.transparent_bg)
        scene.render.image_settings.color_mode = "RGBA" if self.transparent_bg else "RGB"

        samples = max(1, int(self.samples))
        if hasattr(scene, "cycles"):
            scene.cycles.samples = samples
            scene.cycles.seed = int(self.seed)
            if engine_name == "CYCLES":
                self._configure_cycles_device(scene, self.device)
        if hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = samples

    @staticmethod
    def _engine_name(engine: str) -> str:
        if bpy is None:
            return "BLENDER_EEVEE"
        val = str(engine).upper()
        supported = bpy.context.scene.render.bl_rna.properties["engine"].enum_items.keys()
        if val == "CYCLES":
            return "CYCLES"
        if val in supported:
            return val
        if "BLENDER_EEVEE_NEXT" in supported:
            return "BLENDER_EEVEE_NEXT"
        return "BLENDER_EEVEE"

    @staticmethod
    def _normalize_device(device: str) -> str:
        val = str(device).strip().lower()
        return val if val in {"auto", "gpu", "cpu"} else "auto"

    @classmethod
    def _configure_cycles_device(cls, scene, device_mode: str) -> None:
        mode = cls._normalize_device(device_mode)
        if mode == "cpu":
            try:
                scene.cycles.device = "CPU"
            except Exception:
                pass
            return

        if cls._try_enable_cycles_gpu(scene):
            return

        # GPU unavailable or failed; always keep a safe CPU fallback.
        try:
            scene.cycles.device = "CPU"
        except Exception:
            pass

    @staticmethod
    def _try_enable_cycles_gpu(scene) -> bool:
        if bpy is None:
            return False

        try:
            scene.cycles.device = "CPU"
        except Exception:
            return False

        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
        except Exception:
            return False

        try:
            prefs.get_devices()
        except Exception:
            pass

        available = {
            str(getattr(device, "type", "")).upper()
            for device in (getattr(prefs, "devices", None) or [])
            if str(getattr(device, "type", "")).upper() != "CPU"
        }
        preferred_backends = ("OPTIX", "CUDA", "HIP", "METAL", "ONEAPI", "OPENCL")
        backend = next((item for item in preferred_backends if item in available), None)
        if backend is None:
            return False

        try:
            prefs.compute_device_type = backend
        except Exception:
            return False

        try:
            prefs.get_devices()
        except Exception:
            pass

        enabled_gpu = False
        for device in (getattr(prefs, "devices", None) or []):
            device_type = str(getattr(device, "type", "")).upper()
            use_gpu = device_type == backend
            try:
                device.use = use_gpu
            except Exception:
                continue
            enabled_gpu = enabled_gpu or use_gpu

        if not enabled_gpu:
            return False

        try:
            scene.cycles.device = "GPU"
        except Exception:
            return False
        return True
