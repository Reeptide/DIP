"""Unit tests for app/imaging.py: tiling, reconstruction, and the JPEG codec.

No Docker/Kafka/Redis needed - these are pure functions over numpy arrays.
"""
import numpy as np
import pytest

from app.imaging import (
    tile_geometries, split_image_into_tiles, reconstruct_image_from_tiles,
    encode_image, decode_image,
)
from app.config import MAX_TILES


def make_test_image(width, height, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)


def make_compressible_test_image(width, height, seed=0):
    """A smooth gradient + a few solid shapes - like a real photo, JPEG
    compresses this well. Pure per-pixel random noise (make_test_image) is
    the JPEG worst case (no spatial redundancy for DCT to exploit) and is
    fine for shape/dtype tests but not for asserting reconstruction fidelity
    against a real quality=95 encode."""
    import cv2
    rng = np.random.default_rng(seed)
    xv, yv = np.meshgrid(np.linspace(0, 255, width), np.linspace(0, 255, height))
    image = np.stack([xv, yv, (xv + yv) / 2], axis=-1).astype(np.uint8)
    for _ in range(10):
        center = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        radius = int(rng.integers(20, min(width, height) // 6))
        color = tuple(int(c) for c in rng.integers(0, 255, size=3))
        cv2.circle(image, center, radius, color, thickness=-1)
    return image


class TestTileGeometries:
    def test_exact_multiple_of_tile_size(self):
        geoms = tile_geometries(1024, 1024, tile_size=512)
        assert len(geoms) == 4
        # tile_ids are dense, 0-indexed, and assigned in raster order
        assert [g['tile_id'] for g in geoms] == [0, 1, 2, 3]

    def test_covers_the_whole_image_with_no_overlap(self):
        width, height, tile_size = 1300, 900, 512
        geoms = tile_geometries(width, height, tile_size)
        covered = np.zeros((height, width), dtype=bool)
        for g in geoms:
            region = covered[g['y']:g['y'] + g['height'], g['x']:g['x'] + g['width']]
            assert not region.any(), "tiles overlap"
            covered[g['y']:g['y'] + g['height'], g['x']:g['x'] + g['width']] = True
        # Uncovered strip, if any, must be the sub-16px remainder that
        # tile_geometries intentionally drops (see its own docstring).
        uncovered = ~covered
        if uncovered.any():
            ys, xs = np.where(uncovered)
            assert (height - ys.max() - 1) < 16 or (width - xs.max() - 1) < 16

    def test_tiny_remainder_strip_is_dropped_not_a_degenerate_tile(self):
        # 512 + 5px: the trailing 5px strip is below the 16px floor and
        # must be skipped entirely, not emitted as a 5px-wide tile.
        geoms = tile_geometries(517, 512, tile_size=512)
        assert len(geoms) == 1
        assert geoms[0]['width'] == 512

    def test_deterministic_given_same_dimensions(self):
        a = tile_geometries(2048, 1536, tile_size=512)
        b = tile_geometries(2048, 1536, tile_size=512)
        assert a == b


class TestSplitImageIntoTiles:
    def test_tile_count_matches_geometry(self):
        image = make_test_image(1024, 768)
        tiles = split_image_into_tiles(image, tile_size=512)
        assert len(tiles) == len(tile_geometries(1024, 768, 512))

    def test_each_tile_data_matches_its_declared_region(self):
        image = make_test_image(1024, 1024)
        tiles = split_image_into_tiles(image, tile_size=512)
        for t in tiles:
            region = image[t['y']:t['y'] + t['height'], t['x']:t['x'] + t['width']]
            assert np.array_equal(t['tile_data'], region)

    def test_raises_when_max_tiles_exceeded(self):
        # A 1x1 tile_size on a modest image blows well past MAX_TILES.
        image = make_test_image(200, 200)
        with pytest.raises(ValueError):
            split_image_into_tiles(image, tile_size=1)

    def test_tile_data_is_a_copy_not_a_view(self):
        # reconstruct_image_from_tiles round-trips through encode/decode so
        # this doesn't currently bite in production, but split explicitly
        # promises a .copy() - mutating a tile must not corrupt the source.
        image = make_test_image(512, 512)
        tiles = split_image_into_tiles(image, tile_size=512)
        tiles[0]['tile_data'][:] = 0
        assert image.max() > 0


class TestReconstructRoundTrip:
    def test_split_encode_decode_reconstruct_is_lossless_enough(self):
        """The real pipeline: tile -> JPEG-encode each tile -> (network) ->
        JPEG-decode -> place back into a canvas. JPEG is lossy, so this
        can't assert bit-exact equality - it asserts the reconstructed
        image is structurally correct (right shape, tiles land in the
        right place, no black bands / garbage) the way master/app.py's
        own manual verification did each time in this session."""
        width, height = 1024, 768
        image = make_compressible_test_image(width, height, seed=42)
        tiles = split_image_into_tiles(image, tile_size=512)

        tiles_for_reconstruction = [
            {
                'tile_id': t['tile_id'],
                'processed_tile': encode_image(t['tile_data']),
                'x': t['x'], 'y': t['y'],
                'width': t['width'], 'height': t['height'],
            }
            for t in tiles
        ]

        result = reconstruct_image_from_tiles(tiles_for_reconstruction, width, height)

        assert result.shape == (height, width, 3)
        # JPEG quality=95 round-trip: mean absolute pixel difference should
        # be small, not "the image is unrecognizable."
        diff = np.abs(result.astype(int) - image.astype(int))
        assert diff.mean() < 5.0

    def test_missing_tile_leaves_a_black_region_not_a_crash(self):
        # reconstruct_image_from_tiles is intentionally defensive (Step 7's
        # world has partial results mid-flight) - a bad tile should be
        # skipped, not take down reconstruction.
        width, height = 512, 512
        result = reconstruct_image_from_tiles([], width, height)
        assert result.shape == (height, width, 3)
        assert result.max() == 0

    def test_corrupt_tile_bytes_are_skipped_not_fatal(self):
        width, height = 512, 512
        bad_tile = {
            'tile_id': 0, 'processed_tile': b'not a jpeg',
            'x': 0, 'y': 0, 'width': width, 'height': height,
        }
        result = reconstruct_image_from_tiles([bad_tile], width, height)
        assert result.shape == (height, width, 3)


class TestCodec:
    def test_encode_then_decode_round_trips(self):
        image = make_test_image(256, 256)
        decoded = decode_image(encode_image(image))
        assert decoded.shape == image.shape

    def test_decode_rejects_too_small_payload(self):
        with pytest.raises(ValueError):
            decode_image(b'tiny')

    def test_decode_rejects_garbage_payload(self):
        with pytest.raises(Exception):
            decode_image(b'x' * 200)

    def test_encoded_bytes_are_raw_jpeg_not_base64(self):
        # Step 5's whole point: no base64 anywhere in this path anymore.
        image = make_test_image(64, 64)
        data = encode_image(image)
        assert data[:2] == b'\xff\xd8'  # JPEG SOI marker
