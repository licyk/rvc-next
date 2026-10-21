import json
import logging
from typing import Any

import pytest
from typer.testing import CliRunner

from rvc_next.cli.app import get_app
from rvc_next.logger import LOGGER_NAME

EXPECTED_TREE = {
    "analyse": None,
    "assets": {"download": None, "list": None, "verify": None},
    "config": {"get": None, "path": None, "set": None, "show": None},
    "convert": None,
    "doctor": None,
    "effects": None,
    "env": None,
    "jobs": {"list": None, "show": None},
    "live": {"check": None, "devices": None, "latency": None, "run": None, "test-tone": None},
    "model": {
        "catalog": None,
        "download": None,
        "edit": None,
        "export": None,
        "extract": None,
        "import": None,
        "index": {"assign": None, "attach": None, "build": None},
        "info": None,
        "list": None,
        "merge": None,
        "remove": None,
    },
    "output": {"check": None},
    "preset": {"list": None, "remove": None, "save": None},
    "separate": None,
    "train": {"dataset": {"scan": None, "speakers": None}, "export": None, "import": None, "list": None, "new": None, "run": None, "status": None},
    "tts": {"speak": None, "voices": None},
    "version": None,
    "webui": None,
}


def _tree(group):
    import typer

    click_group: Any = typer.main.get_command(group) if not hasattr(group, "commands") else group
    return {name: (_tree(cmd) if hasattr(cmd, "commands") else None) for name, cmd in click_group.commands.items()}


@pytest.fixture
def runner(tmp_path, monkeypatch, tiny_assets_dir):
    monkeypatch.setenv("RVC_NEXT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_NEXT_PATHS__ASSETS_DIR", str(tiny_assets_dir))
    monkeypatch.setenv("RVC_NEXT_COMPUTE__DEVICE", "cpu")
    logger = logging.getLogger(LOGGER_NAME)
    level = logger.level
    yield CliRunner()
    logger.setLevel(level)


def test_command_tree():
    assert _tree(get_app()) == EXPECTED_TREE


def test_debug_everywhere(runner):
    app = get_app()
    for args in (["--debug", "version"], ["model", "--debug", "list"], ["model", "index", "--debug", "--help"]):
        r = runner.invoke(app, args)
        assert r.exit_code == 0, (args, r.output)


def test_version_and_config(runner):
    app = get_app()
    r = runner.invoke(app, ["version", "--json"])
    assert r.exit_code == 0 and json.loads(r.stdout)["rvc_next"]
    assert runner.invoke(app, ["config", "set", "server.port", "9001"]).exit_code == 0
    assert json.loads(runner.invoke(app, ["config", "get", "server.port"]).stdout) == 9001
    r = runner.invoke(app, ["config", "set", "nope.key", "1"])
    assert r.exit_code == 4


def test_model_json_matches_api_shape(runner, tiny_voice_file):
    from fastapi.testclient import TestClient

    from rvc_next.api.app import create_app
    from rvc_next.core.context import build_services

    app = get_app()
    r = runner.invoke(app, ["model", "import", str(tiny_voice_file), "--name", "Cli Voice", "--json"])
    assert r.exit_code == 0, r.output
    listed = json.loads(runner.invoke(app, ["model", "list", "--json"]).stdout)
    services = build_services()
    try:
        with TestClient(create_app(services, serve_ui=False), base_url="http://localhost") as c:
            assert c.get("/api/v1/models").json() == listed
    finally:
        services.close()
    r = runner.invoke(app, ["model", "info", "nope"])
    assert r.exit_code == 2


def test_convert_with_path_voice(runner, tiny_voice_file, wav_file, tmp_path, no_asset_checks):
    app = get_app()
    out = tmp_path / "out"
    r = runner.invoke(app, ["convert", str(wav_file), "-v", str(tiny_voice_file), "--f0", "pm", "-o", str(out), "--json"])
    assert r.exit_code == 0, r.output
    job = json.loads(r.stdout)
    assert job["state"] == "completed"
    assert [p.name for p in out.iterdir()] == ["speech.tiny-voice.wav"]


def test_missing_asset_exit_code(runner, tiny_voice_file, wav_file, tmp_path):
    app = get_app()
    r = runner.invoke(app, ["convert", str(wav_file), "-v", str(tiny_voice_file), "-o", str(tmp_path / "o")])
    assert r.exit_code == 7


def test_data_dir_option_before_or_after_the_command(runner, tmp_path):
    """``--data-dir`` at any level points the command at another data directory, as the environment variable does."""
    app = get_app()
    for args, folder in ((["--data-dir", str(tmp_path / "a"), "config", "path"], "a"), (["config", "path", "--data-dir", str(tmp_path / "b")], "b")):
        r = runner.invoke(app, args)
        assert r.exit_code == 0, r.output
        assert r.output.strip().endswith(str(tmp_path / folder / "settings.toml")), r.output
    r = runner.invoke(app, [*["jobs", "list"], "--data-dir", str(tmp_path / "c")])
    assert r.exit_code == 0, r.output
    assert (tmp_path / "c" / "rvc-next.db").is_file() and not (tmp_path / "data" / "rvc-next.db").exists()


def test_live_device_options_choose_the_driver():
    """A device by name tries its drivers in turn; by a variant's id, or with --input-driver, that driver alone."""
    from pathlib import Path
    from types import SimpleNamespace
    from typing import cast

    import typer

    from rvc_next.cli.commands.live import _find, _with_driver
    from rvc_next.core.context import Services
    from rvc_next.core.live.models import DeviceList, ResolvedDevice
    from rvc_next.engine.audio_io.devices import device_list, resolve_selection

    data = json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "devices" / "windows.json").read_text())
    listing = DeviceList.model_validate(device_list(data["hostapis"], data["devices"], "win32", data["default_hostapi"]))

    def resolve(devices, listing):
        status, device, _ = resolve_selection(devices.input.model_dump(), listing.model_dump(), "input")
        return [ResolvedDevice.model_validate({"role": "input", "status": status, "device": device})]

    services = cast("Services", SimpleNamespace(live=SimpleNamespace(resolve=resolve)))  # only what _with_driver reads
    by_name = _find(listing, "input", "Microphone (Realtek(R) Audio)")
    assert by_name is not None and by_name.driver == "auto" and by_name.host_api == "wasapi"
    ds = next(v for p in listing.inputs for v in p.variants if p.name == "Microphone (Realtek(R) Audio)" and v.host_api == "directsound")
    by_id = _find(listing, "input", ds.id)
    assert by_id is not None and by_id.driver == "fixed" and by_id.host_api == "directsound"
    fixed = _with_driver(services, listing, "input", by_name, "mme")
    assert (fixed.driver, fixed.host_api, fixed.physical_key) == ("fixed", "mme", by_name.physical_key)
    assert _with_driver(services, listing, "input", by_id, "auto").driver == "auto"
    with pytest.raises(typer.BadParameter, match="not asio"):
        _with_driver(services, listing, "input", by_name, "asio")
