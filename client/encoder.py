from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np

from common.types import Codec, EncodedFrame

logger = logging.getLogger(__name__)


class FrameEncoder:
    """Client-side encoder: converts raw camera images to compressed formats for gRPC transport."""

    def __init__(
        self,
        codec: Codec = Codec.JPEG,
        jpeg_quality: int = 85,
        webp_quality: int = 80,
        h264_preset: str = "ultrafast",
    ) -> None:
        self._codec = codec
        self._jpeg_quality = jpeg_quality
        self._webp_quality = webp_quality
        self._h264_preset = h264_preset
        self._jpeg_encoder: Any = None
        self._h264_encoder: Any = None
        self._init_encoders()

    def _init_encoders(self) -> None:
        if self._codec == Codec.JPEG:
            try:
                from turbojpeg import TurboJPEG

                self._jpeg_encoder = TurboJPEG()
                logger.info("TurboJPEG encoder initialized (quality=%d)", self._jpeg_quality)
            except ImportError:
                logger.info("TurboJPEG not available, using PIL JPEG fallback")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self,
        image: np.ndarray,
        camera_id: str = "",
        codec: Codec | None = None,
    ) -> EncodedFrame:
        """Encode a raw image (H, W, C) uint8 into an EncodedFrame."""
        codec = codec or self._codec
        timestamp = time.time_ns()
        h, w = image.shape[:2]
        c = image.shape[2] if image.ndim == 3 else 1

        if codec == Codec.RAW:
            data = image.tobytes()
        elif codec == Codec.JPEG:
            data = self._encode_jpeg(image)
        elif codec == Codec.WEBP:
            data = self._encode_webp(image)
        elif codec == Codec.H264:
            data = self._encode_h264(image)
        else:
            data = image.tobytes()
            codec = Codec.RAW

        return EncodedFrame(
            data=data,
            codec=codec,
            width=w,
            height=h,
            channels=c,
            camera_id=camera_id,
            timestamp_ns=timestamp,
        )

    def encode_batch(
        self,
        images: list[np.ndarray],
        camera_id: str = "",
    ) -> list[EncodedFrame]:
        return [self.encode(img, camera_id=camera_id) for img in images]

    @property
    def codec(self) -> Codec:
        return self._codec

    @codec.setter
    def codec(self, value: Codec) -> None:
        self._codec = value

    # ------------------------------------------------------------------
    # Adaptive quality
    # ------------------------------------------------------------------

    def adapt_quality(self, latency_ms: float, target_ms: float = 20.0) -> None:
        """Lower quality if latency is too high, increase if headroom exists."""
        if latency_ms > target_ms * 1.5:
            self._jpeg_quality = max(30, self._jpeg_quality - 10)
            self._webp_quality = max(20, self._webp_quality - 10)
            logger.debug("Reduced encode quality: jpeg=%d webp=%d", self._jpeg_quality, self._webp_quality)
        elif latency_ms < target_ms * 0.5:
            self._jpeg_quality = min(95, self._jpeg_quality + 5)
            self._webp_quality = min(90, self._webp_quality + 5)

    # ------------------------------------------------------------------
    # Encoders
    # ------------------------------------------------------------------

    def _encode_jpeg(self, image: np.ndarray) -> bytes:
        if self._jpeg_encoder is not None:
            return self._jpeg_encoder.encode(image, quality=self._jpeg_quality)

        from PIL import Image

        img = Image.fromarray(image)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self._jpeg_quality)
        return buf.getvalue()

    def _encode_webp(self, image: np.ndarray) -> bytes:
        from PIL import Image

        img = Image.fromarray(image)
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=self._webp_quality)
        return buf.getvalue()

    def _encode_h264(self, image: np.ndarray) -> bytes:
        """Encode single frame as H.264 NAL unit using PyAV."""
        try:
            import av

            h, w = image.shape[:2]
            codec = av.CodecContext.create("libx264", "w")
            codec.width = w
            codec.height = h
            codec.pix_fmt = "yuv420p"
            codec.time_base = "1/50"
            codec.options = {"preset": self._h264_preset, "tune": "zerolatency"}

            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame = frame.reformat(format="yuv420p")

            packets = codec.encode(frame)
            packets += codec.encode(None)  # flush

            data = b""
            for pkt in packets:
                data += bytes(pkt)
            return data

        except ImportError:
            logger.warning("PyAV not available for H.264 encoding, falling back to JPEG")
            return self._encode_jpeg(image)
