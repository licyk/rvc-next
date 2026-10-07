"""``rvc-next tts``: speech from text, as an input to convert."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Annotated

import typer

from rvc_next.cli.output import console, open_services, print_json


def tts_voices(
    locale: Annotated[str | None, typer.Option("--locale", help="Only voices whose locale starts with this, e.g. en, zh-CN")] = None,
    refresh: Annotated[bool, typer.Option("--refresh", help="Fetch the list again instead of the week-old cache")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Print the voices as JSON")] = False,
) -> None:
    """List the text-to-speech voices."""
    with open_services() as services:
        result = services.tts.voices(refresh)
    if not result.available:
        raise typer.BadParameter("Text to speech needs edge-tts: pip install rvc-next[tts]")
    voices = [v for v in result.voices if not locale or v.locale.lower().startswith(locale.lower())]
    if json_output:
        print_json([v.model_dump() for v in voices])
        return
    for v in voices:
        console.print(f"{v.id}\t{v.gender}\t{v.language}", markup=False, highlight=False)


def tts_speak(
    text: Annotated[str, typer.Argument(help="The text to speak; '-' reads it from standard input")],
    voice: Annotated[str, typer.Option("--voice", help="A voice id from 'rvc-next tts voices', e.g. en-US-AriaNeural")],
    output: Annotated[Path | None, typer.Option("-o", "--output", help="Write the MP3 here (default: print where it was stored)", dir_okay=False)] = None,
    rate: Annotated[int, typer.Option(min=-50, max=100, help="Speaking rate change in percent")] = 0,
    pitch: Annotated[int, typer.Option(min=-50, max=50, help="Pitch change in Hz")] = 0,
) -> None:
    """Speak text with an online voice (the text goes to Microsoft's speech service) and save it as MP3."""
    import sys

    from rvc_next.core.tts.models import TtsRequest

    if text == "-":
        text = sys.stdin.read()
    with open_services() as services:
        f = services.tts.synthesize(TtsRequest(text=text, voice=voice, rate=rate, pitch=pitch))
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(f.path, output)
    console.print(str(output or f.path), markup=False, highlight=False)
