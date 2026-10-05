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
