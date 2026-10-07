"""Device and precision choice, and the cache of loaded models.

The GPU rule is the original's (``configs/config.py``): a GPU under 4 GiB or below SM 5.3 is not
used; SM 6.1 and GTX 16xx cards run fp32; newer cards run fp16; without a usable CUDA device
DirectML is tried, then the CPU. Here it is a function of a requested device instead of an import
side effect, and nothing reads ``sys.argv``.

Beyond the original, as ComfyUI's ``model_management``: AMD cards on a ROCm build of torch answer
to ``torch.cuda`` and are ``cuda:N`` devices (``torch.version.hip`` tells them apart; the SM rule
does not apply, they need 4 GiB and run fp16), and Intel GPUs are ``xpu:N`` (4 GiB, fp16 when the
device has it). One build of torch carries one of CUDA, ROCm or XPU, so the GPU ids of training
(``gpus``) index the devices of :func:`gpu_backend`.
"""

from __future__ import annotations

import gc
import logging
import re
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

    from rvc_next.engine.f0.base import F0Provider
    from rvc_next.engine.index.retrieval import LoadedIndex
    from rvc_next.engine.models.loader import LoadedVoice

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceProfile:
    """One compute device and what the GPU rule made of it."""

    id: str
    name: str
    kind: str
    """cpu, cuda (NVIDIA, or AMD on ROCm), xpu, dml or mps."""
    memory_gb: float = 0.0
    sm: float = 0.0
    eligible: bool = True
    fp16: bool = False
    reason: str | None = None


def cuda_profile(index: int) -> DeviceProfile:
    """Apply the GPU rule to one CUDA device."""
    import torch

    device_id = f"cuda:{index}"
    try:
        major, minor = torch.cuda.get_device_capability(index)
        name = torch.cuda.get_device_name(index)
        mem_bytes = torch.cuda.get_device_properties(index).total_memory
    except Exception as e:  # a broken driver must not stop the CPU path
        return DeviceProfile(device_id, f"CUDA {index}", "cuda", eligible=False, reason=f"Cannot inspect the device: {e}")
    mem_gb = mem_bytes / (1024**3) + 0.4
    if is_rocm():
        # The capability is the gfx version (10.3 for gfx1030), not an SM; every ROCm card runs fp16.
        if mem_gb < 4:
            return DeviceProfile(device_id, name, "cuda", mem_gb, eligible=False, reason="Under 4 GiB of memory")
        return DeviceProfile(device_id, name, "cuda", mem_gb, fp16=True)
    sm = major + minor / 10.0
    is_16_series = bool(re.search(r"16\d{2}", name)) and sm == 7.5
    if mem_gb < 4 or sm < 5.3:
        return DeviceProfile(device_id, name, "cuda", mem_gb, sm, eligible=False, reason="Under 4 GiB of memory or below compute capability 5.3")
    if sm == 6.1 or is_16_series:
        return DeviceProfile(device_id, name, "cuda", mem_gb, sm, fp16=False, reason="Pascal SM 6.1 and GTX 16xx cards run in fp32")
    if sm > 6.1:
        return DeviceProfile(device_id, name, "cuda", mem_gb, sm, fp16=True)
    return DeviceProfile(device_id, name, "cuda", mem_gb, sm, eligible=False, reason="Compute capability below 6.1")


def cuda_profiles() -> list[DeviceProfile]:
    import torch

    if not torch.cuda.is_available():
        return []
    return [cuda_profile(i) for i in range(torch.cuda.device_count())]


def is_rocm() -> bool:
    """Whether torch is a ROCm (HIP) build, whose AMD devices are ``cuda:N``."""
    try:
        import torch.version

        return bool(getattr(torch.version, "hip", None))
    except Exception:
        return False


def xpu_available() -> bool:
    try:
        import torch

        return hasattr(torch, "xpu") and bool(torch.xpu.is_available())
    except Exception:
        return False


def xpu_profile(index: int) -> DeviceProfile:
    """The GPU rule for one Intel XPU device: at least 4 GiB, fp16 when the device has it."""
    import torch

    device_id = f"xpu:{index}"
    try:
        props = torch.xpu.get_device_properties(index)
        name = str(props.name)
        mem_gb = props.total_memory / (1024**3)
        fp16 = bool(getattr(props, "has_fp16", True))
    except Exception as e:
        return DeviceProfile(device_id, f"XPU {index}", "xpu", eligible=False, reason=f"Cannot inspect the device: {e}")
    if mem_gb < 4:
        return DeviceProfile(device_id, name, "xpu", mem_gb, eligible=False, reason="Under 4 GiB of memory")
    return DeviceProfile(device_id, name, "xpu", mem_gb, fp16=fp16, reason=None if fp16 else "The device has no fp16 support")


def xpu_profiles() -> list[DeviceProfile]:
    if not xpu_available():
        return []
    import torch

    return [xpu_profile(i) for i in range(torch.xpu.device_count())]


def gpu_backend() -> str | None:
    """``cuda`` (NVIDIA or ROCm) or ``xpu``: the device type the GPU ids of training refer to."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        return None
    return "xpu" if xpu_available() else None


def gpu_profiles() -> list[DeviceProfile]:
    """The devices of :func:`gpu_backend`, numbered as the GPU ids of training."""
    backend = gpu_backend()
    if backend == "cuda":
        return cuda_profiles()
    if backend == "xpu":
        return xpu_profiles()
    return []


def device_type(device: Any) -> str:
    """cuda, xpu, mps, dml or cpu for a ``torch.device``, a device string or the DirectML device."""
    text = str(getattr(device, "type", None) or device).split(":", 1)[0].lower()
    return "dml" if text in ("privateuseone", "dml") else text


def supports_half(device: Any) -> bool:
    """Whether fp16 may run on ``device`` (the GPU rule decides whether it does)."""
    return device_type(device) in ("cuda", "xpu")


def visible_devices_env(gpus: tuple[int, ...] | list[int], backend: str | None = None) -> dict[str, str]:
    """The environment that shows a child process only ``gpus`` (HIP reads ``CUDA_VISIBLE_DEVICES`` too)."""
    ids = ",".join(map(str, gpus))
    return {"ZE_AFFINITY_MASK": ids} if (backend or gpu_backend()) == "xpu" else {"CUDA_VISIBLE_DEVICES": ids}


def synchronize(device: Any) -> None:
    """Wait for the device's queued work (CUDA, ROCm, XPU); nothing elsewhere."""
    import torch

    kind = device_type(device)
    if kind == "cuda":
        torch.cuda.synchronize(device)
    elif kind == "xpu":
        torch.xpu.synchronize(device)


def directml_device() -> Any | None:
    """The DirectML device, if ``torch_directml`` is installed and actually works."""
    try:
        import torch
        import torch_directml  # ty: ignore[unresolved-import]

        device = torch_directml.device(torch_directml.default_device())
        probe = torch.ones(1, dtype=torch.float32).to(device)
        _ = (probe + 1).cpu()
        return device
    except Exception:
        return None


def mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:
        return False


def torch_versions() -> tuple[str, str | None]:
    """torch's version and the CUDA version it was built with."""
    import torch
    import torch.version

    return torch.__version__, torch.version.cuda


def torch_backend() -> str | None:
    """The accelerator torch was built for: ``CUDA 12.8``, ``ROCm 6.4``, ``XPU``; None for a CPU build."""
    import torch.version

    hip = getattr(torch.version, "hip", None)
    if hip:
        return f"ROCm {hip}"
    if torch.version.cuda:
        return f"CUDA {torch.version.cuda}"
    xpu = getattr(torch.version, "xpu", None)
    if xpu or xpu_available():
        return f"XPU {xpu}" if xpu else "XPU"
    return None


def list_devices() -> list[DeviceProfile]:
    """Every device the application could use, CPU last."""
    out = [*cuda_profiles(), *xpu_profiles()]
    if directml_device() is not None:
        out.append(DeviceProfile("dml", "DirectML", "dml"))
    if mps_available():
        out.append(DeviceProfile("mps", "Apple Metal (experimental)", "mps"))
    out.append(DeviceProfile("cpu", "CPU", "cpu"))
    return out


@dataclass(frozen=True)
class DeviceChoice:
    device: Any
    """A ``torch.device``, or the DirectML device object."""
    id: str
    kind: str
    name: str
    fp16: bool
    memory_gb: float = 0.0
    reason: str | None = None

    @property
    def dtype(self) -> torch.dtype:
        import torch

        return torch.float16 if self.fp16 else torch.float32


def choose_device(requested: str = "auto", precision: str = "auto") -> DeviceChoice:
    """Resolve a ``compute.device`` and ``compute.precision`` pair into a device and dtype."""
    import torch

    requested = (requested or "auto").lower()
    choice: DeviceChoice | None = None
    if requested == "auto":
        eligible = [p for p in gpu_profiles() if p.eligible]
        if eligible:
            best = max(eligible, key=lambda p: (p.sm, p.memory_gb))
            choice = DeviceChoice(torch.device(best.id), best.id, best.kind, best.name, best.fp16, best.memory_gb, best.reason)
        else:
            dml = directml_device()
            if dml is not None:
                choice = DeviceChoice(dml, "dml", "dml", "DirectML", False)
    elif requested.startswith(("cuda", "xpu")):
        kind = "xpu" if requested.startswith("xpu") else "cuda"
        index = int(requested.split(":", 1)[1]) if ":" in requested else 0
        if kind == "cuda":
            count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        else:
            count = torch.xpu.device_count() if xpu_available() else 0
        if index < count:
            profile = cuda_profile(index) if kind == "cuda" else xpu_profile(index)
            # An explicit choice is honoured even when the rule would skip the card, in fp32.
            choice = DeviceChoice(torch.device(profile.id), profile.id, kind, profile.name, profile.fp16 and profile.eligible, profile.memory_gb, profile.reason)
        else:
            choice = DeviceChoice(torch.device("cpu"), "cpu", "cpu", "CPU", False, reason=f"{requested} is not available")
    elif requested == "dml":
        dml = directml_device()
        if dml is not None:
            choice = DeviceChoice(dml, "dml", "dml", "DirectML", False)
        else:
            choice = DeviceChoice(torch.device("cpu"), "cpu", "cpu", "CPU", False, reason="DirectML is not available")
    elif requested == "mps" and mps_available():
        choice = DeviceChoice(torch.device("mps"), "mps", "mps", "Apple Metal", False)
    if choice is None:
        choice = DeviceChoice(torch.device("cpu"), "cpu", "cpu", "CPU", False)
    if precision == "fp32" and choice.fp16:
        choice = DeviceChoice(choice.device, choice.id, choice.kind, choice.name, False, choice.memory_gb, "fp32 requested")
    elif precision == "fp16" and choice.kind in ("cuda", "xpu") and not choice.fp16:
        choice = DeviceChoice(choice.device, choice.id, choice.kind, choice.name, True, choice.memory_gb, "fp16 requested")
    return choice


# -- memory (after ComfyUI's comfy/model_management.py) ------------------------------------


def is_oom(error: BaseException) -> bool:
    """Whether ``error`` means the device ran out of memory.

    ``torch.OutOfMemoryError`` (CUDA, and MPS on recent torch), or an accelerator/runtime error that
    says so: CUDA's ``AcceleratorError`` code 2, MPS's "MPS backend out of memory", DirectML's
    "not enough GPU video memory". Does not import torch when nothing has.
    """
    torch = sys.modules.get("torch")
    if torch is not None:
        oom = getattr(torch, "OutOfMemoryError", None) or getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom is not None and isinstance(error, oom):
            return True
        accel = getattr(torch, "AcceleratorError", None)
        if accel is not None and isinstance(error, accel) and getattr(error, "error_code", None) == 2:
            _discard_cuda_async_error()
            return True
    text = str(error).lower()
    if isinstance(error, (RuntimeError, MemoryError)) and ("out of memory" in text or "not enough gpu video memory" in text):
        if torch is not None and "cuda" in text:
            _discard_cuda_async_error()
        return True
    return False


def _discard_cuda_async_error() -> None:
    """Let a pending asynchronous CUDA error surface and drop it, so the next operation starts clean."""
    try:
        import torch

        if torch.cuda.is_available():
            a = torch.tensor([1], dtype=torch.uint8, device="cuda")
            _ = a + a
            torch.cuda.synchronize()
    except Exception:
        pass


def soft_empty_cache() -> None:
    """Collect garbage, then hand the allocator's cached blocks back to the driver (CUDA, ROCm, XPU, MPS).

    ComfyUI's ``soft_empty_cache``: ``synchronize`` first so no kernel still uses a block, and
    ``ipc_collect`` for memory shared with other processes. Does nothing when torch is not loaded.
    """
    gc.collect()
    torch = sys.modules.get("torch")
    if torch is None:
        return
    try:
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif hasattr(torch, "xpu") and torch.xpu.is_available() and getattr(torch.xpu, "is_initialized", lambda: True)():
            torch.xpu.synchronize()
            torch.xpu.empty_cache()
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception as e:  # a broken context after an error must not stop the unload
        logger.warning("Emptying the GPU cache failed: %s", e)


@dataclass(frozen=True)
class ChunkConfig:
    """Long-audio chunking, in seconds (the original's ``x_pad/x_query/x_center/x_max``)."""

    x_pad: int
    x_query: int
    x_center: int
    x_max: int

    @classmethod
    def for_device(cls, choice: DeviceChoice) -> ChunkConfig:
        if choice.kind in ("cuda", "xpu") and choice.memory_gb > 0 and int(choice.memory_gb) <= 4:
            return cls(1, 5, 30, 32)
        if choice.fp16:
            return cls(3, 10, 60, 65)
        return cls(1, 6, 38, 41)


@dataclass(frozen=True)
class AssetPaths:
    """Where the runtime finds its model files, in the original's ``assets/`` layout."""

    root: Path

    def embedder_dir(self, name: str) -> Path:
        """The folder of a content-feature model (``contentvec`` is ``hubert_base``)."""
        from rvc_next.engine.features.embedders import folder

        return self.root / folder(name)

    @property
    def hubert_dir(self) -> Path:
        return self.root / "hubert_base"

    @property
    def rmvpe(self) -> Path:
        return self.root / "rmvpe" / "rmvpe.pt"

    @property
    def rmvpe_onnx(self) -> Path:
        return self.root / "rmvpe" / "rmvpe.onnx"

    @property
    def fcpe(self) -> Path:
        return self.root / "fcpe" / "fcpe_c_v001.pt"

    def crepe(self, capacity: str) -> Path:
        """torchcrepe's weights: ``full.pth`` or ``tiny.pth``."""
        return self.root / "crepe" / f"{capacity}.pth"

    def pretrained(self, version: str, f0: bool, kind: str, sample_rate: str) -> Path:
        folder = "pretrained_v2" if version == "v2" else "pretrained"
        return self.root / folder / f"{'f0' if f0 else ''}{kind}{sample_rate}.pth"


@dataclass
class _Entry:
    value: Any
    key: tuple
    last_used: float = field(default_factory=time.monotonic)


class Runtime:
    """The device, the dtype, the CUDA Graph switch and the cache of loaded models.

    HuBERT and the F0 models are loaded once; the last ``max_voices`` voices and their indexes stay
    loaded. Everything is safe to call from several threads.
    """

    def __init__(
        self,
        assets: AssetPaths | Path | str,
        device: str = "auto",
        precision: str = "auto",
        cuda_graph: bool = False,
        max_voices: int = 3,
        choice: DeviceChoice | None = None,
    ) -> None:
        from rvc_next.engine.graph import configure_cuda_graph

        self.assets = assets if isinstance(assets, AssetPaths) else AssetPaths(Path(assets))
        self.choice = choice or choose_device(device, precision)
        self.cuda_graph = configure_cuda_graph(self.choice.device, cuda_graph)
        self.chunk = ChunkConfig.for_device(self.choice)
        self.max_voices = max(1, max_voices)
        self._lock = threading.RLock()
        self._hubert: dict[str, _Entry] = {}
        """Content-feature models by embedder name (``contentvec`` is RVC's HuBERT)."""
        self._f0: dict[str, _Entry] = {}
        self._voices: OrderedDict[str, _Entry] = OrderedDict()
        self._indexes: dict[str, _Entry] = {}
        logger.info("Compute: %s (%s), %s%s", self.choice.name, self.choice.id, "fp16" if self.choice.fp16 else "fp32", ", CUDA Graph" if self.cuda_graph else "")

    @property
    def device(self) -> Any:
        return self.choice.device

    @property
    def is_half(self) -> bool:
        return self.choice.fp16

    @property
    def dtype(self) -> torch.dtype:
        return self.choice.dtype

    # -- models -------------------------------------------------------------

    def hubert(self, embedder: str = "contentvec") -> Any:
        """The content-feature model of ``embedder`` (``contentvec``: RVC's HuBERT base), loaded once."""
        from rvc_next.engine.features.embedders import asset_id, known
        from rvc_next.engine.features.hubert import load_hubert

        if not known(embedder):
            raise ValueError(f"Unknown content-feature model: {embedder}")
        with self._lock:
            entry = self._hubert.get(embedder)
            if entry is None:
                entry = _Entry(load_hubert(self.assets.embedder_dir(embedder), self.device, self.is_half, asset_id(embedder)), ("hubert", embedder))
                self._hubert[embedder] = entry
            entry.last_used = time.monotonic()
            return entry.value

    def f0(self, method: str) -> F0Provider:
        """The pitch extractor for ``method`` (``engine.f0.METHODS``)."""
        from rvc_next.engine.f0 import create_provider

        with self._lock:
            entry = self._f0.get(method)
            if entry is None:
                entry = _Entry(create_provider(method, self), (method,))
                self._f0[method] = entry
            entry.last_used = time.monotonic()
            return entry.value

    def voice(self, path: Path | str) -> LoadedVoice:
        """A voice model, loaded once per path and modification time."""
        from rvc_next.engine.models.loader import load_voice

        path = Path(path)
        key = (str(path.resolve()), path.stat().st_mtime_ns)
        with self._lock:
            entry = self._voices.get(key[0])
            if entry is not None and entry.key == key:
                self._voices.move_to_end(key[0])
                entry.last_used = time.monotonic()
                return entry.value
            voice = load_voice(path, self.device, self.is_half)
            self._voices[key[0]] = _Entry(voice, key)
            self._voices.move_to_end(key[0])
            while len(self._voices) > self.max_voices:
                _, old = self._voices.popitem(last=False)
                self._dispose(old.value)
            return voice

    def index(self, path: Path | str) -> LoadedIndex:
        """A retrieval index, loaded once per path and modification time."""
        from rvc_next.engine.index.retrieval import load_index

        path = Path(path)
        key = (str(path.resolve()), path.stat().st_mtime_ns)
        with self._lock:
            entry = self._indexes.get(key[0])
            if entry is None or entry.key != key:
                entry = _Entry(load_index(path), key)
                self._indexes[key[0]] = entry
            entry.last_used = time.monotonic()
            return entry.value

    # -- memory -------------------------------------------------------------

    def cached(self) -> list[str]:
        """Names of the loaded models, for the compute panel."""
        with self._lock:
            out = ["hubert" if name == "contentvec" else f"hubert:{name}" for name in self._hubert]
            out += [f"f0:{m}" for m in self._f0]
            out += [f"voice:{Path(k).name}" for k in self._voices]
            out += [f"index:{Path(k).name}" for k in self._indexes]
            return out

    def release(self) -> None:
        """Unload every model and return GPU memory."""
        with self._lock:
            for entry in self._voices.values():
                self._dispose(entry.value)
            self._voices.clear()
            self._indexes.clear()
            self._f0.clear()
            for entry in self._hubert.values():
                self._dispose(entry.value)
            self._hubert.clear()
        soft_empty_cache()

    def release_idle(self, idle_seconds: float) -> bool:
        """Unload everything when nothing was used for ``idle_seconds``. Return whether anything was freed."""
        with self._lock:
            entries = [e for e in [*self._hubert.values(), *self._f0.values(), *self._voices.values(), *self._indexes.values()] if e is not None]
            if not entries or time.monotonic() - max(e.last_used for e in entries) < idle_seconds:
                return False
        self.release()
        return True

    def vram(self) -> tuple[float, float] | None:
        """(used, total) in MiB on a CUDA, ROCm or XPU device, else None."""
        if self.choice.kind not in ("cuda", "xpu"):
            return None
        import torch

        try:
            free, total = (torch.cuda if self.choice.kind == "cuda" else torch.xpu).mem_get_info(self.device)
        except Exception:
            return None
        return (total - free) / 2**20, total / 2**20

    @staticmethod
    def _dispose(model: Any) -> None:
        from rvc_next.engine.graph import clear_cuda_graph_cache

        target = getattr(model, "net_g", model)
        try:
            clear_cuda_graph_cache(target)
        except Exception:
            pass
