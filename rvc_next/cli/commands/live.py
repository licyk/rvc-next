"""live devices | check | test-tone | run."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Annotated

import typer

from rvc_next.cli.output import console, err_console, open_services, print_json, print_table
from rvc_next.cli.params import ParamOptions, build_params

if TYPE_CHECKING:
    from rvc_next.core.context import Services
    from rvc_next.core.live.models import DeviceList, DeviceSelection

JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON")]


def _find(listing: DeviceList, direction: str, ref: str | None) -> DeviceSelection | None:
    """A device by id (any variant), by name, or ``default``; None keeps the saved one."""
    from rvc_next.core.live.models import DeviceSelection

    if ref is None:
        return None
    if ref.lower() == "default":
        return DeviceSelection()
    physicals = listing.inputs if direction == "input" else listing.outputs
    for p in physicals:
        for v in p.variants:
            if v.id == ref:
                return DeviceSelection(device_id=v.id, physical_key=p.key, name=p.name, host_api=v.host_api)
    wanted = ref.lower()
    matches = [p for p in physicals if wanted == p.name.lower()] or [p for p in physicals if wanted in p.name.lower()]
    if len(matches) != 1:
        names = ", ".join(p.name for p in matches) if matches else "none"
        raise typer.BadParameter(f"{ref!r} matches {len(matches)} {direction} devices ({names}); use an id from 'rvc-next live devices'")
    p = matches[0]
    v = next(x for x in p.variants if x.id == p.recommended_id)
    return DeviceSelection(device_id=v.id, physical_key=p.key, name=p.name, host_api=v.host_api)


def live_devices(refresh: Annotated[bool, typer.Option("--refresh", help="Enumerate again")] = True, json_output: JsonOpt = False) -> None:
    """List the server's audio devices: one row per physical device, its drivers and ids."""
    with open_services() as services:
        listing = services.live.devices(refresh=refresh)
        if json_output:
            print_json(listing)
            return
        for title, physicals in (("Inputs", listing.inputs), ("Outputs", listing.outputs)):
            rows = []
            for p in physicals:
                for v in p.variants:
                    mark = ("default " if p.is_default else "") + ("virtual " if p.is_virtual else "")
                    rows.append(
                        (
                            p.name if v.id == p.variants[0].id else "",
                            mark.strip(),
                            v.host_api + (" (recommended)" if v.id == p.recommended_id else ""),
                            v.id,
                            v.channels,
                            f"{v.default_sample_rate}",
                        )
                    )
            print_table(f"{title} on {listing.host}", ["Device", "", "Driver", "Id", "Ch", "Rate"], rows)
        for e in listing.errors:
            console.print(f"[yellow]{e}[/yellow]")


def live_check(json_output: JsonOpt = False) -> None:
    """Check the saved devices: resolution, format, topology and the estimated latency."""
    with open_services() as services:
        result = services.live.check(services.settings.settings.live.devices)
        if json_output:
            print_json(result)
            return
        for r in result.resolved:
            console.print(f"{r.role:8} {r.status:16} {r.device.name if r.device else '-'}{' via ' + r.device.host_api if r.device else ''}", markup=False)
        for p in result.problems:
            console.print(f"[yellow]{p.role}: {p.message}[/yellow]")
        if result.topology:
            console.print(f"{result.sample_rate} Hz · {result.topology} · estimated latency {result.est_latency_ms} ms")
        if not result.ok:
            raise typer.Exit(8)


def live_test_tone(device: Annotated[str, typer.Argument(help="Output device id, name or 'default'")] = "default") -> None:
    """Play a short chime on an output."""
    from rvc_next.core.live.models import TestToneRequest

    with open_services() as services:
        selection = _find(services.live.devices(refresh=True), "output", device)
        services.live.test_tone(TestToneRequest(role="output", device=selection))
        console.print("Played the test sound")


def live_latency(
    block_ms: Annotated[int | None, typer.Option(help="Block length in ms (default: the saved one)")] = None,
    level_db: Annotated[float, typer.Option(help="Peak level of the test bursts in dBFS", min=-40, max=-3)] = -12.0,
    json_output: JsonOpt = False,
) -> None:
    """Measure the real latency of the saved devices with a loopback: connect the output to the input first."""
    from rvc_next.core.live.models import LatencyTestRequest

    with open_services() as services:
        stream = services.settings.settings.live.stream
        if block_ms is not None:
            stream = stream.model_copy(update={"block_ms": block_ms})
        m = services.live.measure_latency(LatencyTestRequest(stream=stream, level_db=level_db))
        if json_output:
            print_json(m)
        elif m.ok:
            console.print(
                f"Latency {m.latency_ms:.0f} ms: round trip {m.round_trip_ms:.1f} ms (measured) + conversion {m.engine_ms:.0f} ms. "
                f"Estimated {m.estimated_ms:.0f} ms. {m.sample_rate} Hz · {m.topology} · block {m.stream.block_ms} ms · "
                f"{m.detected}/{m.pings} bursts, jitter {m.jitter_ms or 0:.2f} ms, {m.snr_db:.0f} dB over noise"
            )
        else:
            hint = (
                "No test burst came back: connect the output to the input (a cable, a virtual cable's loopback, or speakers near the microphone) and raise the levels"
                if m.reason == "no_signal"
                else "The bursts came back at different times (dropouts or a drifting clock); try a longer block or a duplex device"
            )
            err_console.print(f"[yellow]{hint}[/yellow]")
        if m.clipped and not json_output:
            err_console.print("[yellow]The input clipped: lower --level-db or the input gain[/yellow]")
        if not m.ok:
            raise typer.Exit(1)


def live_run(
    voice: Annotated[str | None, typer.Option("--voice", "-v", help="Voice id or name (default: the last one used)")] = None,
    input_: Annotated[str | None, typer.Option("--input", help="Input device id, name or 'default'")] = None,
    output: Annotated[str | None, typer.Option("--output", help="Output device id, name or 'default'")] = None,
    monitor: Annotated[str | None, typer.Option("--monitor", help="Monitor output id, name, 'default' or 'none'")] = None,
    preset: Annotated[str | None, typer.Option(help="Start from this preset")] = None,
    speaker: ParamOptions.speaker = None,
    pitch: ParamOptions.pitch = None,
    formant: ParamOptions.formant = None,
    f0: ParamOptions.f0 = None,
    index_rate: ParamOptions.index_rate = None,
    protect: ParamOptions.protect = None,
    rms_mix: ParamOptions.rms_mix = None,
    block_ms: Annotated[int | None, typer.Option(help="Block length in ms")] = None,
    crossfade_ms: Annotated[int | None, typer.Option(help="Crossfade in ms")] = None,
    context_ms: Annotated[int | None, typer.Option(help="Extra context in ms")] = None,
    threshold_db: Annotated[float | None, typer.Option(help="Input gate in dB; -60 turns it off")] = None,
    allow_fallback: Annotated[bool, typer.Option("--allow-fallback", help="Start even if the saved output is gone and the default would be used")] = False,
    seconds: Annotated[float | None, typer.Option(help="Stop after this many seconds (default: until Ctrl+C)")] = None,
) -> None:
    """Convert live with the server's devices until Ctrl+C, with a one-line status."""
    from rvc_next.core.live.models import LiveConfig

    with open_services() as services:
        live = services.settings.settings.live
        ref = voice or live.last_voice
        if not ref:
            raise typer.BadParameter("Give --voice")
        v = services.models.resolve(ref, allow_path=False)
        base = services.presets.resolve(preset, v.id).params if preset else (live.last_params if v.id == live.last_voice else services.presets.default_for(v.id).params)
        params = build_params(base, speaker=speaker, pitch=pitch, formant=formant, f0=f0, index_rate=index_rate, protect=protect, rms_mix=rms_mix)
        stream = live.stream.model_copy(
            update={k: val for k, val in {"block_ms": block_ms, "crossfade_ms": crossfade_ms, "context_ms": context_ms, "threshold_db": threshold_db}.items() if val is not None}
        )
        listing = services.live.devices(refresh=True)
        devices = live.devices.model_copy()
        if (sel := _find(listing, "input", input_)) is not None:
            devices.input = sel
        if (sel := _find(listing, "output", output)) is not None:
            devices.output = sel
        if monitor is not None:
            devices.monitor = None if monitor.lower() == "none" else _find(listing, "output", monitor)
        services.live.start(LiveConfig(voice_id=v.id, params=params, stream=stream, devices=devices, allow_output_fallback=allow_fallback))
        _status_loop(services, seconds)


def _status_loop(services: Services, seconds: float | None) -> None:
    from rich.live import Live

    started = time.monotonic()
    try:
        with Live(console=err_console, refresh_per_second=4, transient=True) as view:
            while True:
                state = services.live.state()
                stats = services.live.stats()
                if state.state == "error":
                    break
                if state.state == "stopped" and time.monotonic() - started > 2:
                    break
                load = stats.infer_ms_p95 / stats.block_ms if stats.block_ms else 0.0
                view.update(
                    f"{state.state:11} {state.sample_rate or '-'} Hz {state.topology or ''}  latency {stats.est_latency_ms:.0f} ms  load {'>100%' if load > 1 else f'{load:.0%}'}  "
                    f"in {stats.input_peak_db:6.1f} dB  out {stats.output_peak_db:6.1f} dB  under/over {stats.underruns}/{stats.overruns}"
                )
                if seconds is not None and time.monotonic() - started >= seconds:
                    break
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        state = services.live.state()
        if state.state not in ("stopped", "error"):
            services.live.stop()
            for _ in range(40):
                if services.live.state().state in ("stopped", "error"):
                    break
                time.sleep(0.25)
    final = services.live.state()
    if final.error:
        err_console.print(f"[red]{final.error.get('message')}[/red]")
        raise typer.Exit(8 if final.error.get("code") == "device_unavailable" else 1)
    err_console.print("Stopped")
