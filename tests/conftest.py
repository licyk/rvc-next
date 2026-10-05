from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tiny_assets_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tests.tiny import tiny_assets

    return tiny_assets(tmp_path_factory.mktemp("assets"))


@pytest.fixture(scope="session")
def real_assets_dir() -> Path:
    """The real assets folder, for tests marked ``assets``: ``RVC_NEXT_TEST_ASSETS``."""
    value = os.environ.get("RVC_NEXT_TEST_ASSETS")
    if not value or not (Path(value) / "hubert_base" / "config.json").is_file():
        pytest.skip("RVC_NEXT_TEST_ASSETS does not point at an assets folder with HuBERT")
    return Path(value)


@pytest.fixture
def tiny_runtime(tiny_assets_dir: Path):
    from rvc_next.engine.runtime import Runtime

    return Runtime(tiny_assets_dir, device="cpu", precision="fp32")


@pytest.fixture
def make_services(tmp_path: Path, tiny_assets_dir: Path):
    """Build services in a temporary data directory; tiny assets, CPU."""
    from rvc_next.core.context import build_services

    built = []

    def make(start_background: bool = False, **env: str):
        environ = {"RVC_NEXT_PATHS__ASSETS_DIR": str(tiny_assets_dir), "RVC_NEXT_COMPUTE__DEVICE": "cpu", **env}
        s = build_services(data_dir=tmp_path / "data", environ=environ, start_background=start_background)
        built.append(s)
        return s

    yield make
    for s in built:
        s.close()


@pytest.fixture
def services(make_services):
    return make_services()


@pytest.fixture
def no_asset_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tiny HuBERT does not match the catalog's sizes; let conversions use it anyway."""
    from rvc_next.core.assets.service import AssetService

    monkeypatch.setattr(AssetService, "require", lambda self, ids, what="This": None)


@pytest.fixture
def tiny_voice_file(tmp_path: Path) -> Path:
    from tests.tiny import make_tiny_voice

    return make_tiny_voice(tmp_path / "src" / "tiny-voice.pth")


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    import soundfile as sf

    from tests.tiny import tone

    path = tmp_path / "src" / "speech.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), tone(1.5, sr=22050), 22050)
    return path
