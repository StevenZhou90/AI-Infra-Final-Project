import numpy as np
import pytest

from common.types import Codec
from client.encoder import FrameEncoder


@pytest.fixture
def encoder():
    return FrameEncoder(codec=Codec.JPEG, jpeg_quality=85)


@pytest.fixture
def sample_image():
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


class TestFrameEncoder:
    def test_encode_raw(self, sample_image):
        enc = FrameEncoder(codec=Codec.RAW)
        frame = enc.encode(sample_image, camera_id="cam_0")
        assert frame.codec == Codec.RAW
        assert len(frame.data) == 480 * 640 * 3
        assert frame.width == 640
        assert frame.height == 480
        assert frame.camera_id == "cam_0"

    def test_encode_jpeg(self, encoder, sample_image):
        frame = encoder.encode(sample_image)
        assert frame.codec == Codec.JPEG
        assert len(frame.data) < len(sample_image.tobytes())
        assert frame.width == 640
        assert frame.height == 480

    def test_encode_webp(self, sample_image):
        enc = FrameEncoder(codec=Codec.WEBP)
        frame = enc.encode(sample_image)
        assert frame.codec == Codec.WEBP
        assert len(frame.data) < len(sample_image.tobytes())

    def test_encode_batch(self, encoder, sample_image):
        images = [sample_image, sample_image, sample_image]
        frames = encoder.encode_batch(images, camera_id="batch_cam")
        assert len(frames) == 3
        assert all(f.camera_id == "batch_cam" for f in frames)

    def test_timestamp_set(self, encoder, sample_image):
        frame = encoder.encode(sample_image)
        assert frame.timestamp_ns > 0

    def test_adapt_quality_down(self, encoder, sample_image):
        original_q = encoder._jpeg_quality
        encoder.adapt_quality(latency_ms=50.0, target_ms=20.0)
        assert encoder._jpeg_quality < original_q

    def test_adapt_quality_up(self, encoder, sample_image):
        encoder._jpeg_quality = 50
        encoder.adapt_quality(latency_ms=5.0, target_ms=20.0)
        assert encoder._jpeg_quality > 50

    def test_compression_ratio(self, encoder, sample_image):
        frame = encoder.encode(sample_image)
        raw_size = 480 * 640 * 3
        ratio = len(frame.data) / raw_size
        assert ratio < 0.5  # JPEG should compress well
