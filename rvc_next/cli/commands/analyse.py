"""``rvc-next analyse``: the pitch curve and the facts of an audio file."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from rvc_next.cli.output import console, open_services, print_json


def analyse(
    path: Annotated[Path, typer.Argument(help="An audio file", exists=True, dir_okay=False)],
    csv: Annotated[Path | None, typer.Option("--csv", help="Write the pitch curve as time_s,f0_hz (0 where there is no pitch)")] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print the analysis as JSON (the spectrogram base64-encoded)")] = False,
) -> None:
    """Show an audio file's duration, rate, channels and median pitch; optionally write its pitch curve."""
    with open_services() as services:
        result = services.audio.analyse(path.resolve())
    if csv is not None:
        with open(csv, "w", encoding="utf-8") as f:
            f.write("time_s,f0_hz\n")
            for i, hz in enumerate(result.pitch):
                f.write(f"{i * result.pitch_hop:.2f},{hz}\n")
    if json_output:
        print_json(result)
        return
    voiced = sum(1 for v in result.pitch if v > 0)
    median = f"{result.pitch_median:.1f} Hz" if result.pitch_median else "no pitch"
    console.print(f"{path.name}: {result.duration:.2f} s, {result.sample_rate} Hz, {result.channels} ch, {result.codec or '?'}")
    console.print(f"Pitch: median {median}, {voiced / max(1, len(result.pitch)):.0%} of frames voiced")
    if csv is not None:
        console.print(f"Wrote {csv}")
