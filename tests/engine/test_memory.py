"""Freeing GPU memory and recognising out-of-memory errors (after ComfyUI's model management)."""

import torch

from rvc_next.engine.runtime import is_oom, soft_empty_cache


def test_is_oom_recognises_every_backend_message() -> None:
    assert is_oom(torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB"))
    assert is_oom(RuntimeError("MPS backend out of memory (MPS allocated: 8.00 GB)"))
    assert is_oom(RuntimeError("Could not allocate tensor with 1048576 bytes. There is not enough GPU video memory available!"))
    assert not is_oom(RuntimeError("boom"))
    assert not is_oom(ValueError("the setting says out of memory"))


def test_runtime_release_empties_the_cache(tiny_runtime, tmp_path) -> None:
    from tests.tiny import make_tiny_voice

    tiny_runtime.voice(make_tiny_voice(tmp_path / "a.pth"))
    assert any(m.startswith("voice:") for m in tiny_runtime.cached())
    tiny_runtime.release()
    assert tiny_runtime.cached() == []
    soft_empty_cache()  # a no-op on the CPU, never an error
