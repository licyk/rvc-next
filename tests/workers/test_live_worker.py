import json
import os
import secrets
import subprocess
import sys
import threading
import time
from multiprocessing import Pipe
from multiprocessing.connection import Listener
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from rvc_next.engine.audio_io.devices import group_devices
from rvc_next.engine.audio_io.fake import FakeBackend
from rvc_next.protocol import live as P
from rvc_next.workers.live import AnyConnection, LiveWorker
from tests.tiny import make_tiny_voice


def machine() -> tuple[list[dict], list[dict]]:
    def dev(index: int, name: str, i: int, o: int) -> dict[str, Any]:
        return {"name": name, "index": index, "hostapi": 0, "max_input_channels": i, "max_output_channels": o, "default_samplerate": 16000.0, "supported_rates": [16000, 48000]}

    hostapis = [{"name": "ALSA", "devices": [0, 1, 2], "default_input_device": 0, "default_output_device": 1}]
    return hostapis, [dev(0, "Mic", 1, 0), dev(1, "Speakers", 0, 2), dev(2, "Headphones", 0, 2)]


def devices_by_name() -> dict[str, dict]:
    grouped = group_devices(*machine(), "linux")
    return {v["raw_name"]: v for p in grouped["inputs"] + grouped["outputs"] for v in p["variants"]}


def devices(output: str = "Speakers", monitor: str | None = None) -> dict[str, Any]:
    eps = devices_by_name()
    return {
        "input": {"device": eps["Mic"], "channels": None, "sample_rate": None, "exclusive": False},
        "output": {"device": eps[output], "channels": None, "sample_rate": None, "exclusive": False},
        "monitor": {"device": eps[monitor], "channels": None, "sample_rate": None, "exclusive": False} if monitor else None,
        "monitor_source": "converted",
        "monitor_gain_db": 0.0,
        "output_gain_db": 0.0,
    }


class Client:
    """The core's side of the pipe, collecting events."""

    def __init__(self, conn: AnyConnection) -> None:
        self.conn = conn
        self.events: list[P.Message] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        while True:
            try:
                data = self.conn.recv_bytes()
            except (EOFError, OSError):
                return
            msg = P.decode(data)
            if msg is not None:
                with self._lock:
                    self.events.append(msg)

    def send(self, msg: P.Message) -> None:
        self.conn.send_bytes(P.encode(msg))

    def states(self) -> list[str]:
        with self._lock:
            return [e.state for e in self.events if isinstance(e, P.State)]

    def of(self, cls: type) -> list[Any]:
        with self._lock:
            return [e for e in self.events if isinstance(e, cls)]

    def wait(self, predicate: Any, timeout: float = 30.0) -> bool:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if predicate():
                return True
            time.sleep(0.02)
        return False

    def wait_state(self, state: str, timeout: float = 30.0) -> bool:
        return self.wait(lambda: self.states()[-1:] == [state], timeout)


@pytest.fixture
def worker(tmp_path: Path, tiny_runtime, tiny_assets_dir: Path):
    a, b = Pipe()
    backend = FakeBackend(*machine(), time_scale=2.0)
    request = P.WorkerRequest(address=["127.0.0.1", 0], authkey="00", assets_dir=str(tiny_assets_dir), device="cpu", precision="fp32", cuda_graph=False, backend="fake")
    w = LiveWorker(b, request, backend=backend, runtime=tiny_runtime)
    thread = threading.Thread(target=w.serve, daemon=True)
    thread.start()
    client = Client(a)
    yield w, client, backend, tmp_path
    client.send(P.Shutdown())
    thread.join(timeout=10)
    a.close()


def start_config(voice: Path, **kw: Any) -> dict[str, Any]:
    return {
        "voice_path": str(voice),
        "index_path": None,
        "params": {"f0_method": "pm", "rms_mix_rate": 1.0},
        "stream": {"block_ms": 100, "crossfade_ms": 50, "context_ms": 300},
        "devices": devices(**kw),
    }


def test_start_update_swap_stop(worker) -> None:
    w, client, backend, tmp = worker
    voice = make_tiny_voice(tmp / "a.pth")
    client.send(P.Start(config=start_config(voice)))
    assert client.wait_state("running")
    assert client.states()[:5] == ["stopped", "starting", "loading", "prewarming", "running"]
    running = client.of(P.State)[-1]
    assert running.topology == "duplex" and running.sample_rate == 16000
    assert client.wait(lambda: len(client.of(P.Stats)) >= 3 and client.of(P.Stats)[-1].infer_ms_p50 > 0 and client.of(P.Stats)[-1].input_peak_db > -30)
    stats = client.of(P.Stats)[-1]
    assert stats.block_ms == 100 and stats.est_latency_ms > 200

    engine, net_g = w.engine, w.engine.voice.net_g
    prewarms = []
    real_prewarm = engine.prewarm
    engine.prewarm = lambda: prewarms.append(1) or real_prewarm()
    client.send(P.UpdateVoice(params={"pitch": 4, "speaker_id": 0}))
    assert client.wait(lambda: w.engine.params.pitch == 4)
    client.send(P.UpdateStream(stream={"threshold_db": -50}))
    assert client.wait(lambda: w.engine.stream.threshold_db == -50)
    assert w.session is not None and w.session.block == 1600
    client.send(P.UpdateStream(stream={"block_ms": 200}))
    assert client.wait(lambda: w.session is not None and w.session.block == 3200)
    assert w.engine is engine and w.engine.voice.net_g is net_g and client.states()[-1] == "running"
    assert len(prewarms) == 1  # the new buffers, before the audio reopened; the gate change needed none

    other = make_tiny_voice(tmp / "b.pth", version="v1", seed=4)
    client.send(P.SetVoice(voice_path=str(other), index_path=None, speaker_id=0))
    assert client.wait(lambda: w.engine.voice.path == other)
    assert client.wait(lambda: client.of(P.State)[-1].voice_path == str(other))
    assert client.states()[-1] == "running" and client.of(P.State)[-1].error is None
    assert client.wait(lambda: np.abs(backend.output(1)[-3200:]).max() > 0)

    # A voice that will not load: reported, and the session goes on converting with the one it had.
    client.send(P.SetVoice(voice_path=str(tmp / "missing.pth"), index_path=None, speaker_id=0))
    assert client.wait(lambda: client.of(P.State)[-1].error is not None)
    failed = client.of(P.State)[-1]
    assert failed.state == "running" and failed.voice_path == str(other)
    assert failed.error["code"] == "not_found" and failed.error["detail"]["reason"] == "voice_load"
    assert w.engine.voice.path == other and w.voice_path == str(other)
    produced = backend.output(1).shape[0]
    assert client.wait(lambda: backend.output(1).shape[0] > produced + 3200)

    client.send(P.SetDevices(devices=devices(output="Headphones")))
    assert client.wait(lambda: backend.output(2).shape[0] > 3 * 3200)
    assert client.states()[-1] == "running"

    client.send(P.Stop())
    assert client.wait_state("stopped")
    assert client.states()[-2:] == ["stopping", "stopped"]


def test_device_loss_then_reconnect(worker) -> None:
    w, client, backend, tmp = worker
    client.send(P.Start(config=start_config(make_tiny_voice(tmp / "a.pth"))))
    assert client.wait_state("running")
    backend.lose(0)
    assert client.wait(lambda: len(client.of(P.DeviceLost)) == 1)
    lost = client.of(P.DeviceLost)[0]
    assert lost.direction == "input" and lost.reason == "missing" and lost.device_id == devices_by_name()["Mic"]["id"]
    assert client.wait_state("reconnecting")
    backend.restore(0)
    client.send(P.SetDevices(devices=devices()))
    assert client.wait_state("running")


def test_errors_are_reported_not_fatal(worker) -> None:
    w, client, backend, tmp = worker
    client.send(P.Start(config=start_config(tmp / "missing.pth")))
    assert client.wait(lambda: client.states()[-1:] == ["error"])
    assert client.of(P.State)[-1].error["code"] == "not_found"
    # A refused device.
    backend.refuse[1] = "busy"
    client.send(P.Start(config=start_config(make_tiny_voice(tmp / "a.pth"))))
    assert client.wait(lambda: client.states()[-1:] == ["error"] and client.of(P.State)[-1].error["code"] == "device_unavailable")
    assert client.of(P.State)[-1].error["detail"]["reason"] == "busy"
    backend.refuse.clear()
    # The processor crashing mid-stream.
    client.send(P.Start(config=start_config(make_tiny_voice(tmp / "a.pth"))))
    assert client.wait_state("running")

    def boom(x: Any) -> Any:
        raise RuntimeError("boom")

    w.session.processor = boom
    assert client.wait(lambda: client.states()[-1:] == ["error"])
    assert client.of(P.State)[-1].error["message"] == "boom"
    assert any("boom" in e.message for e in client.of(P.Log))
    # Still serving.
    client.send(P.Stop())
    assert client.wait_state("stopped")


def test_meter_passthrough_and_tone(worker) -> None:
    w, client, backend, tmp = worker
    eps = devices_by_name()
    client.send(P.Meter(on=True, input={"device": eps["Mic"]}))
    assert client.wait(lambda: any(s.input_peak_db > -30 for s in client.of(P.Stats)))
    assert client.of(P.State)[-1].meter
    client.send(P.Meter(on=False))
    assert client.wait(lambda: w.aux is None)
    w.devices = devices(monitor="Headphones")
    client.send(P.Passthrough(on=True))
    assert client.wait(lambda: backend.output(2).size > 0 and np.abs(backend.output(2)).max() > 0.05)
    client.send(P.Passthrough(on=False))
    client.send(P.TestTone(device={"device": eps["Speakers"]}))
    assert client.wait(lambda: any(i == 1 for i, _ in backend.played))


def test_worker_process_end_to_end(tmp_path: Path, tiny_assets_dir: Path) -> None:
    authkey = secrets.token_bytes(16)
    listener = Listener(("127.0.0.1", 0), authkey=authkey)
    hostapis, raw = machine()
    request = {
        "address": list(listener.address),
        "authkey": authkey.hex(),
        "assets_dir": str(tiny_assets_dir),
        "device": "cpu",
        "precision": "fp32",
        "cuda_graph": False,
        "backend": "fake",
        "fake": {"hostapis": hostapis, "devices": raw, "time_scale": 2.0},
    }
    req = tmp_path / "live.json"
    req.write_text(json.dumps(request))
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    proc = subprocess.Popen([sys.executable, "-m", "rvc_next.workers.live", str(req)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        conn = listener.accept()
        client = Client(conn)
        assert client.wait_state("stopped", timeout=60)
        client.send(P.Start(config=start_config(make_tiny_voice(tmp_path / "a.pth"))))
        assert client.wait_state("running", timeout=120)
        assert client.wait(lambda: len(client.of(P.Stats)) >= 2)
        client.send(P.Shutdown())
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
        listener.close()


def test_release_memory_when_idle_and_unload_on_out_of_memory(worker) -> None:
    import torch

    w, client, backend, tmp = worker
    client.send(P.Start(config=start_config(make_tiny_voice(tmp / "a.pth"))))
    assert client.wait_state("running")
    assert any(m.startswith("voice:") for m in client.of(P.State)[-1].cached)
    # Not while the session converts.
    client.send(P.ReleaseMemory())
    assert client.wait(lambda: any("Not freeing" in e.message for e in client.of(P.Log)))
    assert w.engine is not None and client.states()[-1] == "running"
    client.send(P.Stop())
    assert client.wait_state("stopped")
    client.send(P.ReleaseMemory())
    assert client.wait(lambda: w.engine is None and client.of(P.State)[-1].cached == [])
    assert client.states()[-1] == "stopped"

    # Out of memory mid-stream: everything is unloaded and the error says why.
    client.send(P.Start(config=start_config(make_tiny_voice(tmp / "a.pth"))))
    assert client.wait_state("running")

    def oom(x: Any) -> Any:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")

    w.session.processor = oom
    assert client.wait(lambda: client.states()[-1:] == ["error"])
    last = client.of(P.State)[-1]
    assert last.error["code"] == "compute_unavailable" and last.error["detail"] == {"reason": "out_of_memory"}
    assert last.cached == [] and w.engine is None
