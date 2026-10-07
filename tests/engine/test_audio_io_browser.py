import time

import numpy as np

from rvc_next.engine.audio_io.browser import BrowserBackend, browser_device, float_to_pcm16, pcm16_to_float
from rvc_next.engine.audio_io.streams import AudioSession, Endpoint, SessionConfig

RATE = 48000
BLOCK = 4800


def test_the_browser_clock_drives_a_duplex_session() -> None:
    sent: list[bytes] = []
    backend = BrowserBackend(sent.append)
    ep = Endpoint(device=browser_device(RATE))
    session = AudioSession(backend, SessionConfig(input=ep, output=Endpoint(device=browser_device(RATE))), BLOCK, RATE, processor=lambda x: -x)
    assert session.topology == "duplex" and session.sample_rate == RATE
    session.start()
    try:
        t = np.arange(RATE) / RATE
        tone = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        frame = 960  # 20 ms, as the page sends
        for i in range(0, RATE, frame):
            backend.feed(float_to_pcm16(tone[i : i + frame]))
            # the processing thread converts a block once one is in; give it the time a real one would have
            if (i + frame) % BLOCK == 0:
                deadline = time.monotonic() + 1
                while session.processed_blocks < (i + frame) // BLOCK and time.monotonic() < deadline:
                    time.sleep(0.001)
        out = np.concatenate([pcm16_to_float(p) for p in sent])
        assert out.shape[0] == RATE  # as many samples out as in
        # After the prefill (one block of silence) and one block of processing, the inverted tone.
        lag = 2 * BLOCK
        np.testing.assert_allclose(out[lag:], -tone[: RATE - lag], atol=2e-4)
        assert np.abs(out[:BLOCK]).max() == 0
    finally:
        session.stop()
    backend.feed(float_to_pcm16(np.zeros(frame, np.float32)))  # after stop: dropped
    assert sum(len(p) for p in sent) == 2 * RATE


def test_without_portaudio_the_server_devices_refuse() -> None:
    import pytest

    from rvc_next.engine.audio_io.browser import NoDevices
    from rvc_next.engine.audio_io.streams import DeviceOpenError

    backend = NoDevices("PortAudio library not found")
    assert backend.query() == ([], [], 0)
    with pytest.raises(DeviceOpenError) as e:
        AudioSession(backend, SessionConfig(input=Endpoint(), output=Endpoint()), BLOCK, RATE).start()
    assert e.value.reason == "missing" and "PortAudio library not found" in e.value.message
