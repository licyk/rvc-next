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
    "preset": {"list": None, "remove": None, "save": None},
    "separate": None,
    "train": {"dataset": {"scan": None, "speakers": None}, "export": None, "import": None, "list": None, "new": None, "run": None, "status": None},
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
