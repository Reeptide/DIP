"""Tiling, reconstruction, and JPEG codec.

decode_image uses the validating version from the original worker (it
rejected obviously-corrupt payloads); the original master's copy skipped
that check. Unified on the safer one since both master (reconstruction)
and worker (tile processing) now share this module.

encode_image/decode_image work on raw JPEG bytes, not base64 text. Tiles
used to travel base64-encoded as Kafka JSON fields; since Step 5 they live
in MinIO (app/blobstore.py) as raw objects and Kafka only carries the key,
so there is no longer a reason to pay base64's ~33% size overhead anywhere
in the pipeline.
"""
import logging

import cv2
import numpy as np

from app.config import TILE_SIZE, MAX_TILES

logger = logging.getLogger(__name__)


def tile_geometries(width, height, tile_size=TILE_SIZE):
    """Tile layout for an image of this size: [{tile_id, x, y, width, height}].

    Split out of split_image_into_tiles so the layout can be recomputed from
    nothing but the stored original_width/original_height - the Step 7 reaper
    needs a missing tile's geometry to republish its task, long after the
    source image and the in-memory tile list are gone. Both callers share
    this one loop precisely so tile_id numbering can never drift between
    what the master dispatched and what the reaper reconstructs.
    """
    geometries = []

    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)

            if (x_end - x) < 16 or (y_end - y) < 16:
                logger.debug(f"Skipping tiny tile at ({x},{y})")
                continue

            geometries.append({
                'tile_id': len(geometries),
                'x': x,
                'y': y,
                'width': x_end - x,
                'height': y_end - y,
            })

    return geometries


def split_image_into_tiles(image, tile_size=TILE_SIZE):
    """Split image into tiles with safety checks.

    Args:
        image: OpenCV image
        tile_size: Tile size in pixels

    Returns:
        List of tile dicts: {tile_id, x, y, width, height, tile_data}
    """
    height, width = image.shape[:2]

    tiles_x = (width + tile_size - 1) // tile_size
    tiles_y = (height + tile_size - 1) // tile_size
    expected_tiles = tiles_x * tiles_y

    if expected_tiles > MAX_TILES:
        raise ValueError(
            f"Image too large: would create {expected_tiles} tiles (max: {MAX_TILES}). "
            f"Reduce image size or increase tile size."
        )

    logger.info(f"Splitting image {width}x{height} into {tile_size}x{tile_size} tiles (expected: {expected_tiles})")

    tiles = []
    for geom in tile_geometries(width, height, tile_size):
        tile = image[geom['y']:geom['y'] + geom['height'],
                     geom['x']:geom['x'] + geom['width']].copy()
        tiles.append({**geom, 'tile_data': tile})

    logger.info(f"Total tiles created: {len(tiles)}")
    return tiles


def encode_image(image) -> bytes:
    """Encode OpenCV image to raw JPEG bytes."""
    ok, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Failed to encode image")
    return buffer.tobytes()


def decode_image(data: bytes):
    """Decode raw JPEG bytes to an OpenCV image, with validation."""
    try:
        if not data or len(data) < 100:
            raise ValueError("Invalid or too small image data")

        nparr = np.frombuffer(data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None or image.size == 0:
            raise ValueError("Failed to decode image or empty image")

        return image

    except Exception as e:
        logger.error(f"Image decode error: {str(e)}")
        raise


def reconstruct_image_from_tiles(tiles_data, original_width, original_height):
    """Reconstruct the final image from processed tiles.

    tiles_data[i]['processed_tile'] must already be raw JPEG bytes (fetched
    from MinIO by the caller) - this function doesn't know about blob
    storage, only about decoding bytes and placing them in the canvas.
    """
    logger.info(f"Reconstructing image {original_width}x{original_height} from {len(tiles_data)} tiles")

    reconstructed = np.zeros((original_height, original_width, 3), dtype=np.uint8)

    for tile_info in tiles_data:
        try:
            tile_image = decode_image(tile_info['processed_tile'])
            x = tile_info['x']
            y = tile_info['y']
            width = tile_info['width']
            height = tile_info['height']

            reconstructed[y:y + height, x:x + width] = tile_image
        except Exception as e:
            logger.error(f"Error reconstructing tile {tile_info.get('tile_id')}: {e}")

    logger.info("Image reconstruction completed")
    return reconstructed
