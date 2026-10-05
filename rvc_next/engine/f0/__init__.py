"""Pitch extraction: one interface over pm, RMVPE and FCPE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rvc_next.engine.f0.base import F0Provider

if TYPE_CHECKING:
    from rvc_next.engine.runtime import Runtime

METHODS = ("pm", "rmvpe", "fcpe")


def create_provider(method: str, runtime: Runtime) -> F0Provider:
    if method == "pm":
        from rvc_next.engine.f0.pm import PmProvider

        return PmProvider()
    if method == "rmvpe":
        from rvc_next.engine.f0.rmvpe_provider import RmvpeProvider

        return RmvpeProvider(runtime.assets.rmvpe, runtime.device, runtime.is_half, runtime.assets.rmvpe_onnx)
    if method == "fcpe":
        from rvc_next.engine.f0.fcpe import FcpeProvider

        return FcpeProvider(runtime.assets.fcpe, runtime.device)
    raise ValueError(f"Unsupported F0 method: {method}")


__all__ = ["METHODS", "F0Provider", "create_provider"]
