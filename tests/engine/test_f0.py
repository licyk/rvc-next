import numpy as np

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
