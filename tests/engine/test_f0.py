import numpy as np
import pytest

from rvc_next.engine.f0.base import interpolate_unvoiced, shift, to_coarse


def test_interpolate_fills_gaps_and_keeps_edges() -> None:
    f0 = np.array([0, 100, 0, 0, 160, 0], dtype=np.float64)
    out = interpolate_unvoiced(f0.copy())
    assert out.tolist() == [100, 100, 120, 140, 160, 160]


def test_all_unvoiced_stays_zero() -> None:
    assert interpolate_unvoiced(np.zeros(4)).tolist() == [0, 0, 0, 0]


def test_shift_octave() -> None:
    assert shift(np.array([110.0]), 12).tolist() == [220.0]


def test_coarse_bounds() -> None:
    coarse = to_coarse(np.array([0.0, 50.0, 1100.0, 5000.0]))
    assert coarse.tolist()[0] == 1 and coarse.tolist()[1] == 1 and coarse.tolist()[2] == 255 and coarse.tolist()[3] == 255
    assert coarse.dtype == np.int32


class _Fixed:
    name = "fixed"

    def __init__(self, f0: list[float]) -> None:
        self.f0 = np.array(f0, dtype=np.float64)

    def compute(self, x: np.ndarray, p_len: int) -> np.ndarray:
        return self.f0.copy()


def test_offline_f0_reports_voicing_and_keeps_zeros_on_request() -> None:
    from rvc_next.engine.f0.base import offline_f0

    provider = _Fixed([0, 200, 0, 200, 0])
    coarse, hz, voiced = offline_f0(provider, np.zeros(800), 5, 0)
    assert voiced.tolist() == [False, True, False, True, False]
    assert hz.min() == 200  # filled: the original's rule
    coarse, hz, voiced = offline_f0(provider, np.zeros(800), 5, 12, unvoiced="zero")
    assert hz.tolist() == [0, 400, 0, 400, 0] and coarse[0] == 1 and coarse[1] > 1


def test_high_register_fixes_octave_errors() -> None:
    from rvc_next.engine.f0.high_register import fix_high_register

    normal = np.array([300.0, 600.0, 400.0, 0.0, 0.0, 700.0])
    guide = np.array([300.0, 1200.0, 1200.0, 1100.0, 300.0, 1400.0])
    out = fix_high_register(normal, guide, "true_pitch", 1250.0)
    # kept (agrees) · doubled · twelfth error → guide · filled · too low to fill · above the ceiling: kept
    assert out.tolist() == [300.0, 1200.0, 1200.0, 1100.0, 0.0, 700.0]
    folded = fix_high_register(normal, guide, "fold", 1250.0)
    assert folded.tolist() == [300.0, 600.0, 600.0, 550.0, 0.0, 700.0]
    assert fix_high_register(normal, guide, "off").tolist() == normal.tolist()


def test_rmvpe_chunks_long_audio_seamlessly(monkeypatch) -> None:
    import rvc_next.engine.f0.rmvpe_provider as rp

    class Frames:
        """Stands in for RMVPE: one value per 160-sample frame, read from the audio itself."""

        def infer_from_audio(self, x: np.ndarray, thred: float) -> np.ndarray:
            n = x.shape[0] // 160 + 1
            padded = np.pad(x, (0, n * 160 - x.shape[0] + 1))
            return padded[::160][:n].astype(np.float64)

    provider = rp.RmvpeProvider.__new__(rp.RmvpeProvider)
    provider.model = Frames()  # ty: ignore[invalid-assignment]
    x = np.arange(160 * 1000 + 37, dtype=np.float64)
    whole = provider.compute(x, 0)
    monkeypatch.setattr(rp, "CHUNK_FRAMES", 128)
    monkeypatch.setattr(rp, "OVERLAP_FRAMES", 20)
    assert np.array_equal(provider.compute(x, 0), whole)


@pytest.mark.assets
def test_rmvpe_high_register_on_a_real_model(real_assets_dir) -> None:
    from rvc_next.engine.f0.rmvpe_provider import RmvpeProvider
    from tests.tiny import tone

    provider = RmvpeProvider(real_assets_dir / "rmvpe" / "rmvpe.pt", "cpu", False)
    x = tone(1.0, freq=1200)
    stock = provider.compute(x, 0)
    fixed = provider.high_register(x, stock, "true_pitch", 1250.0)
    assert abs(np.median(stock[stock > 0]) - 600) < 20 and abs(np.median(fixed[fixed > 0]) - 1200) < 20
