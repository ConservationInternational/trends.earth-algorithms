"""The numpy fallbacks in counterbalancing_numba must match the loop versions."""

import pytest

np = pytest.importorskip("numpy")

from te_algorithms.gdal.land_deg import counterbalancing_numba as cb

NODATA = cb.NODATA_VALUE[0]
MASK = cb.MASK_VALUE[0]


@pytest.fixture
def block():
    rng = np.random.default_rng(7)
    shape = (40, 50)
    status = rng.choice(
        np.array([1, 2, 3, 4, 5, 6, 7, 0, 9, NODATA, MASK], dtype=np.int16), shape
    )
    land_type = rng.choice(
        np.array([11, 12, 21, 22, -5, NODATA], dtype=np.int32), shape
    )
    cell_area = rng.random(shape)
    mask = rng.random(shape) < 0.2
    return status, land_type, cell_area, mask


def _as_dict(d):
    return {
        (tuple(int(x) for x in k) if isinstance(k, tuple) else int(k)): float(v)
        for k, v in dict(d).items()
    }


def _assert_dicts_close(result, expected):
    result, expected = _as_dict(result), _as_dict(expected)
    assert result.keys() == expected.keys()
    for key in expected:
        assert result[key] == pytest.approx(expected[key])


def test_classify_gains_losses(block):
    status, _, _, mask = block
    expected = cb._classify_gains_losses_numba(status, mask)
    result = cb._classify_gains_losses_numpy(status, mask)
    assert result.dtype == np.int16
    np.testing.assert_array_equal(result, expected)


def test_zonal_gains_losses(block):
    exp_gains, exp_losses = cb._zonal_gains_losses_numba(*block)
    gains, losses = cb._zonal_gains_losses_numpy(*block)
    _assert_dicts_close(gains, exp_gains)
    _assert_dicts_close(losses, exp_losses)


def test_zonal_class_breakdown(block):
    expected = cb._zonal_class_breakdown_numba(*block)
    result = cb._zonal_class_breakdown_numpy(*block)
    _assert_dicts_close(result, expected)


def test_zonal_class_breakdown_transition_codes(block):
    # counterbalancing.py encodes baseline/period transitions as 10-32
    _, land_type, cell_area, mask = block
    rng = np.random.default_rng(3)
    trans = rng.choice(np.array([10, 11, 22, 32, NODATA], dtype=np.int16), mask.shape)
    expected = cb._zonal_class_breakdown_numba(trans, land_type, cell_area, mask)
    result = cb._zonal_class_breakdown_numpy(trans, land_type, cell_area, mask)
    _assert_dicts_close(result, expected)


def test_all_masked_returns_empty(block):
    status, land_type, cell_area, mask = block
    mask = np.ones_like(mask)
    gains, losses = cb._zonal_gains_losses_numpy(status, land_type, cell_area, mask)
    assert gains == {} and losses == {}
    assert cb._zonal_class_breakdown_numpy(status, land_type, cell_area, mask) == {}
