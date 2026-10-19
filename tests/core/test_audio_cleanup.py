"""Start-up cleanup of old uploads: an upload an output still uses survives, wherever the data directory is opened from."""

import os
import shutil
import time
from pathlib import Path

import pytest

from rvc_next.core.context import build_services

TWO_DAYS = 2 * 24 * 3600


def _age(folder: Path) -> None:
    then = time.time() - TWO_DAYS
    os.utime(folder, (then, then))


@pytest.mark.parametrize("opened", ["same", "moved", "linked"])
def test_cleanup_keeps_uploads_outputs_use(opened, tmp_path, tiny_assets_dir, wav_file):
    environ = {"RVC_NEXT_PATHS__ASSETS_DIR": str(tiny_assets_dir), "RVC_NEXT_COMPUTE__DEVICE": "cpu"}
    first = tmp_path / "first" / "data"
    s = build_services(data_dir=first, environ=environ)
    used = s.audio.upload("used.wav", [wav_file.read_bytes()])
    unused = s.audio.upload("unused.wav", [wav_file.read_bytes()])
    result = tmp_path / "result.wav"
    shutil.copyfile(wav_file, result)
    s.audio.add_output(None, result, "converted", "result", source_path=Path(used.path), publish=False)
    s.close()

    if opened == "same":
        data = first
    elif opened == "moved":
        data = tmp_path / "second" / "data"
        data.parent.mkdir()
        shutil.move(str(first), str(data))
    else:
        data = tmp_path / "link"
        data.symlink_to(first, target_is_directory=True)
    uploads = (data / "uploads").resolve()
    for folder in uploads.iterdir():
        _age(folder)

    s = build_services(data_dir=data, environ=environ)
    try:
        s.audio.cleanup()
        assert (uploads / used.id).is_dir()
        assert s.audio.get_file(used.id).id == used.id
        assert not (uploads / unused.id).exists()
        with pytest.raises(Exception, match="Unknown audio file"):
            s.audio.get_file(unused.id)
    finally:
        s.close()


def test_cleanup_keeps_recent_uploads(services, wav_file):
    upload = services.audio.upload("new.wav", [wav_file.read_bytes()])
    services.audio.cleanup()
    assert services.audio.get_file(upload.id).id == upload.id
    assert Path(upload.path).is_file()
