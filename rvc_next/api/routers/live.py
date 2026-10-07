"""Live conversion with the server's audio devices, or the browser's. Control is REST; the socket
carries state and stats; ``/browser-audio`` (a WebSocket) carries the browser's audio both ways."""

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from rvc_next.api.deps import ServicesDep
from rvc_next.api.errors import ERROR_RESPONSES
from rvc_next.core.context import Services
from rvc_next.core.live.models import (
    DeviceCheck,
    DeviceList,
    LatencyMeasurement,
    LatencyTestRequest,
    LiveConfig,
    LiveDevices,
    LiveState,
    LiveVoiceRequest,
    RecordingRequest,
    TestToneRequest,
    Toggle,
)
from rvc_next.core.params import StreamParamsModel, VoiceParamsModel

router = APIRouter(prefix="/v1/live", tags=["live"], responses=ERROR_RESPONSES)


@router.get("/state", operation_id="get_live_state")
def get_state(services: ServicesDep) -> LiveState:
    return services.live.state()


@router.post("/start", operation_id="start_live")
def start(services: ServicesDep, body: LiveConfig) -> LiveState:
    return services.live.start(body)


@router.post("/stop", operation_id="stop_live")
def stop(services: ServicesDep) -> LiveState:
    return services.live.stop()


@router.patch("/voice", operation_id="update_live_voice")
def update_voice(services: ServicesDep, body: VoiceParamsModel) -> LiveState:
    """Pitch, formant, index strength and the rest apply at the next block."""
    return services.live.update_voice(body)


@router.patch("/stream", operation_id="update_live_stream")
def update_stream(services: ServicesDep, body: StreamParamsModel) -> LiveState:
    """Block, crossfade and context rebuffer without reloading the voice."""
    return services.live.update_stream(body)


@router.put("/voice-model", operation_id="set_live_voice_model")
def set_voice_model(services: ServicesDep, body: LiveVoiceRequest) -> LiveState:
    return services.live.set_voice(body.voice_id)


@router.put("/devices", operation_id="set_live_devices")
def set_devices(services: ServicesDep, body: LiveDevices) -> LiveState:
    """Save the devices; while running, only the affected streams reopen."""
    return services.live.set_devices(body)


@router.get("/devices", operation_id="list_live_devices")
def list_devices(services: ServicesDep, refresh: bool = False) -> DeviceList:
    return services.live.devices(refresh=refresh)


@router.post("/devices/check", operation_id="check_live_devices")
def check_devices(services: ServicesDep, body: LiveDevices) -> DeviceCheck:
    return services.live.check(body)


@router.post("/devices/test-tone", operation_id="play_test_tone")
def test_tone(services: ServicesDep, body: TestToneRequest) -> LiveState:
    return services.live.test_tone(body)


@router.post("/devices/latency-test", operation_id="measure_live_latency")
def measure_latency(services: ServicesDep, body: LatencyTestRequest) -> LatencyMeasurement:
    """Measure the real latency with a loopback: test bursts go out of the output and must reach the
    input (a cable, a virtual cable's loopback, or speakers near the microphone). Takes a few
    seconds; Live must be stopped."""
    return services.live.measure_latency(body)


@router.post("/meter", operation_id="set_live_meter")
def meter(services: ServicesDep, body: Toggle) -> LiveState:
    """Stream the selected input's level whenever Live is idle, for the next 15 seconds (ask again to
    keep it; ``on: false`` ends it). Returns at once; the meter opens in the background, follows the
    selected input and retries a device that is unavailable."""
    return services.live.meter(body.on)


@router.post("/recording", operation_id="start_live_recording")
def start_recording(services: ServicesDep, body: RecordingRequest) -> LiveState:
    """Record the running session to a WAV file in the outputs folder; it is listed as an output
    (kind ``recording``) once it stops."""
    return services.live.start_recording(body)


@router.delete("/recording", operation_id="stop_live_recording")
def stop_recording(services: ServicesDep) -> LiveState:
    return services.live.stop_recording()


@router.post("/passthrough", operation_id="set_live_passthrough")
def passthrough(services: ServicesDep, body: Toggle) -> LiveState:
    """Hear yourself: pass the input straight to the monitor (or the output)."""
    return services.live.passthrough(body.on)


BROWSER_QUEUE = 100
"""Converted frames waiting for the browser (about two seconds); past it the oldest are dropped."""


@router.websocket("/browser-audio")
async def browser_audio(websocket: WebSocket) -> None:
    """The browser's audio for a session started with ``LiveConfig.browser``: binary messages of mono
    16-bit little-endian PCM at that rate, microphone samples in and converted samples out, as many
    out as in once the session runs."""
    services: Services = websocket.app.state.services
    await websocket.accept()
    loop = asyncio.get_running_loop()
    outgoing: asyncio.Queue[bytes] = asyncio.Queue(maxsize=BROWSER_QUEUE)

    def put(pcm: bytes) -> None:
        if outgoing.full():
            outgoing.get_nowait()
        outgoing.put_nowait(pcm)

    def send(pcm: bytes) -> None:  # on the supervisor's reader thread
        with contextlib.suppress(RuntimeError):  # the loop has closed
            loop.call_soon_threadsafe(put, pcm)

    async def writer() -> None:
        while True:
            await websocket.send_bytes(await outgoing.get())

    link = services.live.open_browser_link(send)
    task = asyncio.create_task(writer())
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data:
                link.feed(data)
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        link.close()
