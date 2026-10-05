"""version, env, doctor."""

from __future__ import annotations

import os
import platform
import shutil
from importlib import metadata
from typing import Annotated, Any

import typer

from rvc_next.cli.output import console, print_json, print_table
from rvc_next.version import VERSION

COMPONENTS = (
    "torch",
    "transformers",
    "faiss-cpu",
    "librosa",
    "praat-parselmouth",
    "av",
    "soundfile",
    "sounddevice",
    "pymss",
    "fastapi",
    "uvicorn",
    "pydantic",
    "typer",
)

ENV_VARS = {
    "RVC_NEXT_DATA_DIR": "Data directory holding settings.toml, the database, models, experiments and outputs",
    "RVC_NEXT_CONFIG_DIR": "Folder for settings.toml, when it should not sit in the data directory",
    "RVC_NEXT_LOG_LEVEL": "Log level: DEBUG, INFO, WARNING or ERROR",
    "RVC_NEXT_<GROUP>__<FIELD>": "Override one setting, e.g. RVC_NEXT_SERVER__PORT=8000 or RVC_NEXT_COMPUTE__DEVICE=cpu",
}


def _components() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in COMPONENTS:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def version(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Show the version of rvc-next and its main components."""
    components = _components()
    if json_output:
        print_json({"rvc_next": VERSION, "python": platform.python_version(), "components": components})
        return
    print_table(None, ["Component", "Version"], [("rvc-next", VERSION), ("python", platform.python_version()), *((k, v or "not installed") for k, v in components.items())])


def env(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """List the environment variables rvc-next reads, and those currently set."""
    in_use = {k: ("***" if "TOKEN" in k else v) for k, v in sorted(os.environ.items()) if k.startswith("RVC_NEXT_")}
    if json_output:
        print_json({"known": ENV_VARS, "set": in_use})
        return
    print_table("Environment variables", ["Name", "Meaning"], ENV_VARS.items())
    if in_use:
        print_table("Set now", ["Name", "Value"], in_use.items())


def _device_line(d: dict[str, Any]) -> str:
    parts = [str(d["name"])]
    if d.get("memory_mb"):
        parts.append(f"{d['memory_mb'] / 1024:.1f} GiB")
    parts.append(str(d["precision"]))
    if not d.get("eligible"):
        parts.append("not used")
    if d.get("reason"):
        parts.append(str(d["reason"]))
    return " · ".join(parts)


def doctor(json_output: Annotated[bool, typer.Option("--json", help="Print JSON")] = False) -> None:
    """Check Python, torch and CUDA, DirectML, ffmpeg, audio devices, the assets and the GPU rule."""
    from rvc_next.cli.output import open_services

    report: dict[str, Any] = {"rvc_next": VERSION, "python": platform.python_version(), "platform": platform.platform(), "components": _components()}
    problems: list[str] = []
    with open_services() as services:
        report["data_dir"] = str(services.settings.data_dir)
        try:
            info = services.compute.info()
            report["compute"] = info.model_dump(mode="json")
        except Exception as e:  # torch missing or broken: report, do not crash
            report["compute"] = {"error": str(e)}
            problems.append(f"torch: {e}")
        report["ffmpeg"] = shutil.which("ffmpeg")
        audio = services.live.backend_info()
        report["sounddevice"] = audio
        if audio.get("error"):
            problems.append(f"Audio devices: {audio['error']} (Live needs them)")
        assets = services.assets.list_assets()
        report["assets"] = [a.model_dump(mode="json") for a in assets]
        for a in assets:
            if a.id in ("hubert", "rmvpe") and a.state != "installed":
                problems.append(f"{a.title} is {a.state}: rvc-next assets download {a.id}")
            if a.state == "corrupt":
                problems.append(f"{a.title} failed its checksum: rvc-next assets download {a.id}")
    report["problems"] = problems
    if json_output:
        print_json(report)
        return
    compute: dict[str, Any] = report["compute"]
    rows: list[tuple[str, str]] = [("rvc-next", VERSION), ("Python", report["python"]), ("Platform", report["platform"]), ("Data", report["data_dir"])]
    if "error" in compute:
        rows.append(("Compute", f"[red]{compute['error']}[/red]"))
    else:
        rows.append(("torch", f"{compute['torch_version']} (CUDA {compute['cuda_version'] or 'none'})"))
        rows.append(("Selected", f"{compute['selected']} · {compute['precision']}"))
        for d in compute["devices"]:
            rows.append((f"  {d['id']}", _device_line(d)))
    rows.append(("ffmpeg", report["ffmpeg"] or "not found (PyAV decodes without it)"))
    sd = report["sounddevice"]
    rows.append(("Audio devices", sd.get("portaudio") or f"[red]{sd.get('error')}[/red]"))
    for a in report["assets"]:
        rows.append((f"Asset {a['id']}", a["state"] + ("" if a["verified"] is None else (" · verified" if a["verified"] else " · checksum mismatch"))))
    print_table("rvc-next doctor", ["Check", "Result"], rows)
    for p in problems:
        console.print(f"[yellow]• {p}[/yellow]")
    if not problems:
        console.print("[green]No problems found[/green]")
