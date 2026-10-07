"""LiveService with the fake PortAudio backend and a tiny voice: devices, check, start, hot updates, stop."""

import threading
import time

import pytest

from rvc_next.core.errors import DeviceError
from rvc_next.core.events.models import LiveStateEvent, LiveStatsEvent
from rvc_next.core.live.models import DeviceSelection, LiveConfig, LiveDevices
from rvc_next.core.params import StreamParamsModel, VoiceParamsModel


def wait_for(predicate, timeout=60.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.fixture
def live(services, no_asset_checks):
    from rvc_next.engine.audio_io.fake import default_devices

    hostapis, devices = default_devices()
    services.live.backend = "fake"
    services.live.fake = {"hostapis": hostapis, "devices": devices, "time_scale": 1.0}
    yield services.live
    services.live.close()


def test_devices_and_check(live):
    listing = live.devices(refresh=True)
    assert {p.name for p in listing.inputs} == {"Fake Microphone"}
    cable = next(p for p in listing.outputs if p.is_virtual)
    sel = DeviceSelection(device_id=cable.recommended_id, physical_key=cable.key, name=cable.name)
    result = live.check(LiveDevices(input=DeviceSelection(), output=sel))
    assert result.ok and result.topology in ("duplex", "split") and result.est_latency_ms
    gone = DeviceSelection(device_id="nope", physical_key="usb mic", name="USB Mic")
    result = live.check(LiveDevices(input=DeviceSelection(), output=gone))
    assert not result.ok and result.problems[0].reason == "fallback"


def test_device_enumerations_never_overlap(live, monkeypatch):
    """The page's poll, its checks and the meter ask at once: one subprocess runs at a time, and the
    callers that waited share the list enumerated after they asked."""
    real = live._run_devices
    running: list[int] = []
    overlap: list[int] = []

    def slow(request, **kw):
        running.append(1)
        overlap.append(len(running))
        time.sleep(0.3)
        try:
            return real(request, **kw)
        finally:
            running.pop()

    monkeypatch.setattr(live, "_run_devices", slow)
    first = threading.Thread(target=live.devices, kwargs={"refresh": True})
    first.start()
    time.sleep(0.1)
    waiting = [threading.Thread(target=live.devices, kwargs={"refresh": True}) for _ in range(5)] + [threading.Thread(target=live.devices)]
    for t in waiting:
        t.start()
    for t in [first, *waiting]:
        t.join(30)
    assert max(overlap) == 1
    assert len(overlap) == 2  # the first, then one for every refresh asked while it ran


def test_check_suggests_another_device_for_a_refused_channel_count(live):
    mic = next(d for d in live.fake["devices"] if d["name"] == "Fake Microphone")
    live.fake["refuse"] = {mic["index"]: "channels"}
    result = live.check(LiveDevices())
    assert not result.ok
    assert [(p.role, p.reason, p.action) for p in result.problems] == [("input", "channels", "choose")]


def test_session(live, services, tiny_voice_file):
    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    stats: list = []
    services.events.subscribe(lambda e: stats.append(e) if isinstance(e, LiveStatsEvent) else None)
    gone = DeviceSelection(device_id="nope", physical_key="usb out", name="USB Out")
    with pytest.raises(DeviceError):
        live.start(LiveConfig(voice_id=voice.id, devices=LiveDevices(output=gone)))
    params = VoiceParamsModel(f0_method="pm", index_rate=0, rms_mix_rate=0)
    state = live.start(LiveConfig(voice_id=voice.id, params=params, stream=StreamParamsModel(block_ms=200, context_ms=500)))
    assert state.state == "starting"
    assert wait_for(lambda: live.state().state == "running"), live.state()
    assert wait_for(lambda: len(stats) > 3)
    live.update_voice(params.model_copy(update={"pitch": 3}))
    live.update_stream(StreamParamsModel(block_ms=300, context_ms=500))
    time.sleep(1.0)
    assert live.state().state == "running"
    assert services.settings.settings.live.last_voice == voice.id and services.settings.settings.live.stream.block_ms == 300
    live.stop()
    assert wait_for(lambda: live.state().state == "stopped")
    # The idle worker keeps its models; Free GPU memory unloads them (it was refused while running).
    assert wait_for(lambda: any(m.startswith("live:voice:") for m in services.compute.usage().cached))
    assert services.compute.usage().can_release
    services.compute.release()
    assert wait_for(lambda: not any(m.startswith("live:") for m in services.compute.usage().cached))


def test_voice_switch_while_running(live, services, tmp_path, monkeypatch):
    """Switching voices hot-swaps in the worker; the state follows the voice the worker reports."""
    import threading
    from pathlib import Path

    from tests.tiny import make_tiny_voice

    def add(name: str, seed: int, version: str = "v2"):
        return services.models.import_paths([make_tiny_voice(tmp_path / name / f"{name}.pth", version=version, speakers=2, seed=seed)], name=name).voices[0]

    a, b, c = add("alpha", 1), add("beta", 2, "v1"), add("gamma", 3)
    params = VoiceParamsModel(f0_method="pm", index_rate=0, rms_mix_rate=0)
    live.start(LiveConfig(voice_id=a.id, params=params, stream=StreamParamsModel(block_ms=200, context_ms=500)))
    assert wait_for(lambda: live.state().state == "running"), live.state()

    live.set_voice(b.id)
    assert live.state().voice_id == b.id and live.state().config.voice_id == b.id
    time.sleep(1.0)
    assert live.state().state == "running" and live.state().voice_id == b.id and live.state().error is None

    # The web UI sends a switch and the speaker reset it causes at once. A slow library lookup widens
    # the window in which, unserialised, the speaker change re-sent the old voice after the new one.
    get = services.models.get

    def slow_get(voice_id):
        time.sleep(0.3)
        return get(voice_id)

    monkeypatch.setattr(services.models, "get", slow_get)
    switch = threading.Thread(target=lambda: live.set_voice(a.id))
    speaker = threading.Thread(target=lambda: live.update_voice(params.model_copy(update={"speaker_id": 1})))
    switch.start()
    time.sleep(0.1)
    speaker.start()
    switch.join()
    speaker.join()
    monkeypatch.setattr(services.models, "get", get)
    time.sleep(1.5)
    state = live.state()
    assert state.voice_id == a.id and state.config.voice_id == a.id and state.config.params.speaker_id == 1 and state.error is None

    # A voice that will not load: the session keeps converting with the voice it had, and says so.
    Path(services.models.get(c.id).model_path).unlink()
    live.set_voice(c.id)
    assert wait_for(lambda: (live.state().error or {}).get("detail", {}).get("reason") == "voice_load"), live.state()
    state = live.state()
    assert state.state == "running" and state.voice_id == a.id and state.config.voice_id == a.id
    assert services.settings.settings.live.last_voice == a.id
    live.stop()
    assert wait_for(lambda: live.state().state == "stopped")


def _loopbacks_listed(live) -> bool:
    return any(p.is_loopback for p in live.devices().inputs)


def _pick(physicals, name: str) -> DeviceSelection:
    p = next(p for p in physicals if p.name == name)
    return DeviceSelection(device_id=p.recommended_id, physical_key=p.key, name=p.name)


def test_show_all_devices_lists_outputs_as_loopback_inputs(live, services):
    """live.show_all_devices lists every output device in the input menu too, recorded as a loopback;
    nothing can play into an input device, so the output menu keeps its own."""
    from rvc_next.core.events.models import DevicesChangedEvent

    events: list = []
    services.events.subscribe(lambda e: events.append(e) if isinstance(e, DevicesChangedEvent) else None)
    listing = live.devices(refresh=True)
    output_names = {p.name for p in listing.outputs}
    assert not any(p.is_loopback for p in listing.inputs)

    services.settings.update({"live": {"show_all_devices": True}})
    assert wait_for(lambda: _loopbacks_listed(live))
    assert events and any(p.is_loopback for p in events[-1].devices.inputs)
    merged = live.devices()
    loopbacks = [p for p in merged.inputs if p.is_loopback]
    assert {p.name for p in loopbacks} == output_names and all(p.key.startswith("loopback:") and p.direction == "input" for p in loopbacks)
    assert merged.inputs[0].name == "Fake Microphone" and {p.name for p in merged.outputs} == output_names
    # The system default input is still the microphone.
    default_in = next(r for r in live.resolve(LiveDevices()) if r.role == "input")
    assert default_in.device is not None and default_in.device.loopback_of is None

    services.settings.update({"live": {"show_all_devices": False}})
    assert wait_for(lambda: not _loopbacks_listed(live))


def test_recording_an_output_device(live, services, tiny_voice_file):
    """A loopback input converts what another output plays; recording the output the voice plays on is refused."""
    services.settings.update({"live": {"show_all_devices": True}})
    assert wait_for(lambda: _loopbacks_listed(live))
    listing = live.devices()
    speakers_in = _pick([p for p in listing.inputs if p.is_loopback], "Fake Speakers")
    speakers, headphones = _pick(listing.outputs, "Fake Speakers"), _pick(listing.outputs, "Fake Headphones")

    result = live.check(LiveDevices(input=speakers_in, output=speakers))
    assert not result.ok and [(p.role, p.reason, p.action) for p in result.problems] == [("input", "feedback", "choose")]
    assert [p.reason for p in live.check(LiveDevices(input=speakers_in)).problems] == ["feedback"]  # the default output is the speakers
    result = live.check(LiveDevices(input=speakers_in, output=headphones))
    assert result.ok and result.topology == "split"

    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    params = VoiceParamsModel(f0_method="pm", index_rate=0, rms_mix_rate=0)
    with pytest.raises(DeviceError) as info:
        live.start(LiveConfig(voice_id=voice.id, params=params, devices=LiveDevices(input=speakers_in, output=speakers)))
    assert info.value.detail["reason"] == "feedback"
    services.settings.update({"live": {"devices": LiveDevices(input=speakers_in, output=speakers).model_dump()}})
    with pytest.raises(DeviceError):
        live.passthrough(True)

    live.start(LiveConfig(voice_id=voice.id, params=params, stream=StreamParamsModel(block_ms=200, context_ms=500), devices=LiveDevices(input=speakers_in, output=headphones)))
    assert wait_for(lambda: live.state().state == "running"), live.state()
    with pytest.raises(DeviceError):
        live.set_devices(LiveDevices(input=speakers_in, output=speakers))
    live.stop()
    assert wait_for(lambda: live.state().state == "stopped")


def test_latency_measurement(live, services):
    """A loopback measurement through the devices worker: the fake cable's 60 ms, the device's block
    and the one-block prefill, then the engine's crossfade + 10 ms; it follows the settings it was taken with."""
    from rvc_next.core.live.models import LatencyTestRequest

    live.fake = {**live.fake, "time_scale": 4.0, "loopback": {"delay_ms": 60.0, "gain": 0.5, "noise": 0.001}}
    stream = StreamParamsModel(block_ms=50, crossfade_ms=50)
    live.update_stream(stream)  # the state keeps a measurement only while it fits the current settings
    m = live.measure_latency(LatencyTestRequest())
    assert m.ok and m.topology == "duplex" and m.detected == m.pings == 4
    assert m.round_trip_ms == 160.0 and m.engine_ms == 60.0 and m.latency_ms == 220.0 and not m.clipped
    assert m.estimated_ms == 50 + 60 + 50 + 20  # block, engine, prefill, the fake's reported 10 + 10 ms
    assert live.state().latency_test == m

    # Settings saved while stopped: the crossfade only changes the engine's part; the block voids it.
    live.update_stream(StreamParamsModel(block_ms=50, crossfade_ms=30))
    test = live.state().latency_test
    assert test is not None and test.engine_ms == 40.0 and test.latency_ms == 200.0
    live.update_stream(StreamParamsModel(block_ms=60, crossfade_ms=30))
    assert live.state().latency_test is None

    live.fake = {k: v for k, v in live.fake.items() if k != "loopback"}
    m = live.measure_latency(LatencyTestRequest(stream=stream))
    assert not m.ok and m.reason == "no_signal" and m.latency_ms is None and m.engine_ms == 60.0


def test_start_reports_every_step(live, services, tiny_voice_file):
    """The state says "starting" before the slow steps (device check, worker start), then follows the
    worker through each loading step; a failed check puts the state back as it was."""
    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    seen: list[tuple[str, str | None]] = []
    services.events.subscribe(lambda e: seen.append((e.state.state, e.state.stage)) if isinstance(e, LiveStateEvent) else None)

    gone = DeviceSelection(device_id="nope", physical_key="usb out", name="USB Out")
    with pytest.raises(DeviceError):
        live.start(LiveConfig(voice_id=voice.id, devices=LiveDevices(output=gone)))
    assert seen == [("starting", "devices"), ("stopped", None)]
    assert live.state().state == "stopped" and live.state().stage is None and live.state().error is None

    seen.clear()
    params = VoiceParamsModel(f0_method="pm", index_rate=0, rms_mix_rate=0)
    live.start(LiveConfig(voice_id=voice.id, params=params, stream=StreamParamsModel(block_ms=200, context_ms=500)))
    assert wait_for(lambda: live.state().state == "running"), live.state()
    assert seen[:3] == [("starting", "devices"), ("starting", "worker"), ("starting", None)]
    loading = [stage for state, stage in seen if state == "loading"]
    assert loading[:2] == ["runtime", "voice"] and "hubert" in loading and "pitch" in loading
    assert seen[-1] == ("running", None)
    live.stop()
    assert wait_for(lambda: live.state().state == "stopped")


def test_start_reuses_a_recent_device_list(live, services, tiny_voice_file, monkeypatch):
    """Start resolves against a list enumerated in the last 10 s (the Live page refreshes every 5 s)
    and enumerates again only when the list is older."""
    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    actions: list[str] = []
    real = live._run_devices
    monkeypatch.setattr(live, "_run_devices", lambda request, **kw: actions.append(request["action"]) or real(request, **kw))
    config = LiveConfig(voice_id=voice.id, params=VoiceParamsModel(f0_method="pm"), stream=StreamParamsModel(block_ms=200, context_ms=500))

    live.devices(refresh=True)  # what the Live page's poll does
    actions.clear()
    live.start(config)
    assert wait_for(lambda: live.state().state == "running")
    assert "enumerate" not in actions
    live.stop()
    assert wait_for(lambda: live.state().state == "stopped")

    live._devices_at -= 60  # a list a minute old
    live.start(config)
    assert wait_for(lambda: live.state().state == "running")
    assert actions.count("enumerate") == 1
    live.stop()
    assert wait_for(lambda: live.state().state == "stopped")


def test_recording_becomes_an_output(live, services, tiny_voice_file):
    import soundfile as sf

    from rvc_next.core.live.models import RecordingRequest

    voice = services.models.import_paths([tiny_voice_file]).voices[0]
    params = VoiceParamsModel(f0_method="pm", index_rate=0, rms_mix_rate=0)
    live.start(LiveConfig(voice_id=voice.id, params=params, stream=StreamParamsModel(block_ms=200, context_ms=500, phase_vocoder=True)))
    assert wait_for(lambda: live.state().state == "running"), live.state()
    state = live.start_recording(RecordingRequest(source="both"))
    assert state.recording is not None and state.recording.path.endswith(".both.wav")
    time.sleep(1.0)
    # Gains alone apply to the running session without reopening its devices.
    live.set_devices(services.settings.settings.live.devices.model_copy(update={"input_gain_db": 6.0}))
    time.sleep(0.5)
    assert live.state().state == "running" and live.state().recording is not None
    live.stop_recording()
    assert wait_for(lambda: any(o.kind == "recording" for o in services.audio.list_outputs().items))
    out = next(o for o in services.audio.list_outputs().items if o.kind == "recording")
    data, rate = sf.read(out.path)
    assert data.ndim == 2 and data.shape[1] == 2 and data.shape[0] > rate // 2 and out.model_id == voice.id
    live.stop()
    assert wait_for(lambda: live.state().state == "stopped")
