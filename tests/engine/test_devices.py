"""The GPU rule on ROCm and Intel XPU builds of torch, with the device queries faked."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.version

from rvc_next.engine import graph, runtime
from rvc_next.engine.runtime import ChunkConfig, choose_device, cuda_profile, device_type, gpu_backend, list_devices, supports_half, torch_backend, visible_devices_env

GIB = 1024**3


@pytest.fixture
def rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    cards = [("AMD Radeon RX 7900 XTX", 24 * GIB, (11, 0)), ("AMD Radeon R7 240", 2 * GIB, (8, 0))]
    monkeypatch.setattr(torch.version, "hip", "6.4.43482", raising=False)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(cards))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i: cards[i][0])
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: cards[i][2])
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(total_memory=cards[i][1]))
    monkeypatch.setattr(runtime, "xpu_available", lambda: False)
    monkeypatch.setattr(runtime, "directml_device", lambda: None)
    monkeypatch.setattr(runtime, "mps_available", lambda: False)


@pytest.fixture
def xpu(monkeypatch: pytest.MonkeyPatch) -> None:
    cards = [SimpleNamespace(name="Intel(R) Arc(TM) A770 Graphics", total_memory=16 * GIB, has_fp16=True)]
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "xpu", "20250101", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "device_count", lambda: len(cards))
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda i: cards[i])
    monkeypatch.setattr(runtime, "directml_device", lambda: None)
    monkeypatch.setattr(runtime, "mps_available", lambda: False)


@pytest.mark.usefixtures("rocm")
def test_rocm_cards_are_cuda_devices_in_fp16_with_a_memory_floor() -> None:
    assert torch_backend() == "ROCm 6.4.43482"
    assert gpu_backend() == "cuda"
    big, small = cuda_profile(0), cuda_profile(1)
    assert (big.kind, big.eligible, big.fp16, big.sm) == ("cuda", True, True, 0.0)
    assert not small.eligible and small.reason == "Under 4 GiB of memory"
    choice = choose_device("auto")
    assert (choice.id, choice.kind, choice.fp16) == ("cuda:0", "cuda", True)
    assert choose_device("cuda:0", "fp32").fp16 is False
    assert visible_devices_env([1]) == {"CUDA_VISIBLE_DEVICES": "1"}


@pytest.mark.usefixtures("rocm")
def test_cuda_graphs_stay_off_on_rocm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph, "detect_cuda_graph_support", lambda device: pytest.fail("ROCm must not probe"))
    assert graph.configure_cuda_graph(torch.device("cuda", 0), True) is False


@pytest.mark.usefixtures("xpu")
def test_intel_xpu_is_chosen_and_listed() -> None:
    from rvc_next.engine.train.experiment import default_batch_size
    from rvc_next.engine.train.loop import training_is_half

    assert torch_backend() == "XPU 20250101"
    assert gpu_backend() == "xpu"
    assert [d.id for d in list_devices()] == ["xpu:0", "cpu"]
    choice = choose_device("auto")
    assert (choice.id, choice.kind, choice.fp16, choice.device) == ("xpu:0", "xpu", True, torch.device("xpu", 0))
    assert ChunkConfig.for_device(choice) == ChunkConfig(3, 10, 60, 65)
    assert choose_device("xpu:0", "fp32").fp16 is False
    assert choose_device("xpu:3").kind == "cpu"
    assert visible_devices_env([0, 1]) == {"ZE_AFFINITY_MASK": "0,1"}
    assert training_is_half((0,))
    assert default_batch_size() == 8


@pytest.mark.usefixtures("xpu")
def test_xpu_without_fp16_or_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    cards = [SimpleNamespace(name="iGPU", total_memory=8 * GIB, has_fp16=False), SimpleNamespace(name="tiny", total_memory=1 * GIB, has_fp16=True)]
    monkeypatch.setattr(torch.xpu, "device_count", lambda: len(cards))
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda i: cards[i])
    profiles = runtime.xpu_profiles()
    assert (profiles[0].eligible, profiles[0].fp16) == (True, False)
    assert not profiles[1].eligible
    assert choose_device("xpu:0", "fp16").fp16 is True


def test_device_types() -> None:
    assert device_type(torch.device("xpu", 1)) == "xpu"
    assert device_type("cuda:0") == "cuda"
    assert device_type("privateuseone:0") == "dml"
    assert supports_half("xpu:0") and supports_half(torch.device("cuda")) and not supports_half("cpu") and not supports_half("mps")


def test_separation_on_xpu_loads_on_the_cpu_and_moves_in_fp32(monkeypatch: pytest.MonkeyPatch) -> None:
    import pymss

    from rvc_next.engine.separate.runner import PymssSeparator

    class Net:
        def __init__(self) -> None:
            self.moved_to = None

        def to(self, device):
            self.moved_to = device
            return self

    class FakeSeparator:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.device = kwargs["device"]
            self.model = Net()
            self.config = SimpleNamespace(training=SimpleNamespace(use_amp=True))

    monkeypatch.setattr(pymss, "MSSeparator", FakeSeparator)
    model = {"model_type": "bs_roformer", "ckpt": "m.ckpt", "config": "m.yaml", "batch_size": 2, "overlap_size": 4}
    sep: Any = PymssSeparator(model, torch.device("xpu", 0), "xpu", half=False).separator
    assert sep.kwargs["device"] == "cpu" and sep.kwargs["inference_params"]["use_amp"] is False
    assert sep.model.moved_to == torch.device("xpu", 0)
    assert sep.device == "xpu:0"
