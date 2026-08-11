import numpy as np

from taxembed.eval.radial import initialization_depth_norm_r


def test_linear_schedule_floor_is_essentially_one():
    depths = np.repeat(np.arange(0, 20), 5)
    r = initialization_depth_norm_r(depths, max_depth=20, radial_schedule="linear")
    assert r > 0.999


def test_log_schedule_floor_is_high_but_below_one():
    depths = np.repeat(np.arange(0, 41), 10)
    r = initialization_depth_norm_r(depths, max_depth=40, radial_schedule="log")
    assert 0.7 < r < 1.0


def test_constant_depths_return_nan_rather_than_raising():
    depths = np.full(50, 7)
    r = initialization_depth_norm_r(depths, max_depth=40, radial_schedule="log")
    assert np.isnan(r)
