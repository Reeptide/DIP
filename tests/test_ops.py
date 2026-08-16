"""Unit tests for app/ops.py: the 13 OpenCV operations and their catalog."""
import numpy as np
import pytest

from app.ops import PROCESSING_OPERATIONS, OPERATION_MAP


def make_test_image(size=64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_catalog_and_map_have_matching_keys():
    # PROCESSING_OPERATIONS drives the upload UI's valid-operation check
    # (master/app.py); OPERATION_MAP is what actually executes. A key in
    # one but not the other is a silent dispatch failure waiting to happen.
    assert set(PROCESSING_OPERATIONS.keys()) == set(OPERATION_MAP.keys())


def test_thirteen_operations_are_registered():
    # The original code's comment said "12 operations" for a 13-entry dict
    # (app/ops.py's own docstring documents the fix) - pin the real count
    # so a silent removal doesn't go unnoticed.
    assert len(OPERATION_MAP) == 13


@pytest.mark.parametrize("op_name", list(OPERATION_MAP.keys()))
def test_every_operation_returns_a_valid_bgr_image_of_the_same_size(op_name):
    image = make_test_image(size=64)
    result = OPERATION_MAP[op_name](image)
    assert result is not None
    assert result.shape[:2] == image.shape[:2]
    assert result.ndim == 3 and result.shape[2] == 3
    assert result.dtype == np.uint8


def test_grayscale_actually_removes_color():
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    image[:, :, 2] = 255  # pure red
    result = OPERATION_MAP['grayscale'](image)
    # After gray->BGR round-trip, all three channels must be equal.
    assert np.array_equal(result[:, :, 0], result[:, :, 1])
    assert np.array_equal(result[:, :, 1], result[:, :, 2])


def test_color_inversion_is_its_own_inverse():
    image = make_test_image(size=32)
    twice = OPERATION_MAP['color_inversion'](OPERATION_MAP['color_inversion'](image))
    assert np.array_equal(twice, image)


def test_edge_canny_on_blank_image_finds_no_edges():
    blank = np.full((64, 64, 3), 128, dtype=np.uint8)
    result = OPERATION_MAP['edge_canny'](blank)
    assert result.max() == 0


def test_threshold_binary_is_actually_binary():
    image = make_test_image(size=32)
    result = OPERATION_MAP['threshold_binary'](image)
    unique = np.unique(result)
    assert set(unique.tolist()).issubset({0, 255})


def test_operations_tolerate_a_minimum_size_tile():
    # tile_geometries floors tiles at 16px - operations must not crash on
    # the smallest real input the pipeline can hand them.
    tiny = make_test_image(size=16)
    for op_name, fn in OPERATION_MAP.items():
        result = fn(tiny)
        assert result.shape[:2] == (16, 16), f"{op_name} changed a 16x16 tile's shape"
