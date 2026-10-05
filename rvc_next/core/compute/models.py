"""Compute device records."""

from typing import Literal

from pydantic import Field

from rvc_next.core.record import Record


class ComputeDevice(Record):
    id: str
    """cpu, cuda:N (NVIDIA, or AMD on ROCm), xpu:N, dml or mps."""
    name: str
    kind: Literal["cpu", "cuda", "xpu", "dml", "mps"]
    memory_mb: float | None = None
    sm: float | None = None
    eligible: bool
    precision: Literal["fp16", "fp32"]
    reason: str | None = None
    """Why the device is not eligible, or why it runs in fp32."""


class ComputeInfo(Record):
    selected: str
    precision: Literal["fp16", "fp32"]
    devices: list[ComputeDevice]
    torch_version: str | None = None
    cuda_version: str | None = None
    backend: str | None = None
    """What torch was built for: ``CUDA 12.8``, ``ROCm 6.4``, ``XPU``; None for a CPU build."""
    cuda_graph: bool = False


class ComputeUsage(Record):
    device: str
    vram_used_mb: float | None = None
    vram_total_mb: float | None = None
    cached: list[str] = Field(default_factory=list)
    """Loaded models; ``live:…`` ones are in the live worker."""
    lease_holder: str | None = None
    busy: list[str] = Field(default_factory=list)
    """What is working now (``jobs``, ``live``); memory is freed only when nothing is."""
    can_release: bool = True
