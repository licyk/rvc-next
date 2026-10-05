import pytest

from rvc_next.core.errors import ValidationError
from rvc_next.core.settings import SettingsService


def test_defaults_and_save(tmp_path):
    s = SettingsService(data_dir=tmp_path, environ={})
    assert s.settings.server.port == 7868
    assert s.settings.live.last_params.index_rate == 0.0
    assert s.settings.convert.default_params.rms_mix_rate == 0.25
    s.update({"server": {"port": 9000}, "compute": {"device": "cpu"}})
    again = SettingsService(data_dir=tmp_path, environ={})
    assert again.settings.server.port == 9000 and again.settings.compute.device == "cpu"


def test_environment_overrides_and_resolved_paths(tmp_path):
    s = SettingsService(data_dir=tmp_path, environ={"RVC_NEXT_COMPUTE__GPU_JOBS": "3", "RVC_NEXT_PATHS__MODELS_DIR": str(tmp_path / "m")})
    assert s.settings.compute.gpu_jobs == 3
    assert s.models_dir == (tmp_path / "m").resolve()
    assert s.outputs_dir == tmp_path.resolve() / "outputs"
    assert "RVC_NEXT_COMPUTE__GPU_JOBS" in s.view().env_overrides


def test_view_hides_the_access_token(tmp_path):
    s = SettingsService(data_dir=tmp_path, environ={})
    s.update({"server": {"access_token": "secret-token"}})
    assert "secret-token" not in s.view().model_dump_json()
    assert s.view().server.access_token_configured is True


def test_invalid_values_are_refused(tmp_path):
    s = SettingsService(data_dir=tmp_path, environ={})
    with pytest.raises(ValidationError):
        s.update({"live": {"stream": {"block_ms": 5}}})
    with pytest.raises(ValidationError):
        s.set_value("nope.field", "1")


def test_browse_roots_reach_every_drive(tmp_path, monkeypatch, services):
    """Without configured roots the file browser offers every drive, not only the one home is on."""
    import rvc_next.core.settings.service as settings_service

    drives = [tmp_path / "D", tmp_path / "E"]
    for d in drives:
        (d / "music").mkdir(parents=True)
        (d / "music" / "take.wav").write_bytes(b"RIFF")
    monkeypatch.setattr(settings_service, "volume_roots", lambda: drives)
    roots = services.settings.browse_roots()
    assert all(d.resolve() in roots for d in drives)

    listing = services.audio.browse()
    assert {str(d.resolve()) for d in drives} <= {e.path for e in listing.entries}
    inside = services.audio.browse(str(drives[1] / "music"))
    assert [e.name for e in inside.entries] == ["take.wav"] and inside.parent == str(drives[1].resolve())
    # A file on another drive passes the server's path check.
    from rvc_next.core.audio.models import AudioRef

    services.audio.check_refs([AudioRef(kind="path", path=str(drives[0] / "music" / "take.wav"))])

    # Configured roots still decide alone.
    services.settings.update({"paths": {"browse_roots": [str(drives[0])]}})
    assert services.settings.browse_roots() == [drives[0].resolve()]


def test_windows_drives_from_the_drive_list():
    from pathlib import Path

    from rvc_next.core.files import windows_drives

    assert windows_drives(lambda: ["C:\\", "D:\\", "E:\\"]) == [Path("C:\\"), Path("D:\\"), Path("E:\\")]


def test_changes_made_by_another_process_are_picked_up_and_kept(tmp_path, monkeypatch):
    """A running server sees `rvc-next config set` (another process writing settings.toml), and its own
    next save merges with that change instead of overwriting it."""
    import rvc_next.core.settings.service as settings_service

    monkeypatch.setattr(settings_service, "FILE_CHECK_INTERVAL", 0.0)
    server = SettingsService(data_dir=tmp_path, environ={})
    changes: list[list[str]] = []
    server.on_keys_change(changes.append)
    cli = SettingsService(data_dir=tmp_path, environ={})
    cli.set_value("compute.gpu_jobs", "3")
    assert server.settings.compute.gpu_jobs == 3 and changes == [["compute.gpu_jobs"]]
    cli.set_value("live.show_meters", "false")
    server.update({"server": {"port": 8123}})  # the web UI saves another field
    reread = SettingsService(data_dir=tmp_path, environ={}).settings
    assert reread.compute.gpu_jobs == 3 and reread.live.show_meters is False and reread.server.port == 8123
    assert changes[-1] == ["live.show_meters", "server.port"]
    # A file that no longer parses is reported and ignored; the settings in use stay.
    (tmp_path / "settings.toml").write_text("this is [not toml", encoding="utf-8")
    assert server.settings.server.port == 8123
