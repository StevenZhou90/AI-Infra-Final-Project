from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np

from common.types import Codec, EncodedFrame
from server.config import VideoConfig

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Server-side decoding of encoded frames (JPEG, WebP, H.264/H.265)."""

    def __init__(self, config: VideoConfig) -> None:
        self._config = config
        self._jpeg_decoder: Any = None
        self._init_decoders()

    def _init_decoders(self) -> None:
        try:
            from turbojpeg import TurboJPEG

            self._jpeg_decoder = TurboJPEG()
            logger.info("TurboJPEG decoder initialized")
        except ImportError:
            logger.info("TurboJPEG not available, falling back to PIL for JPEG")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decode_frame(self, frame: EncodedFrame) -> np.ndarray:
        """Decode an encoded frame to a numpy array (H, W, C) uint8."""
        if frame.codec == Codec.RAW:
            return self._decode_raw(frame)
        elif frame.codec == Codec.JPEG:
            return self._decode_jpeg(frame)
        elif frame.codec == Codec.WEBP:
            return self._decode_pil(frame)
        elif frame.codec in (Codec.H264, Codec.H265):
            return self._decode_video(frame)
        else:
            raise ValueError(f"Unsupported codec: {frame.codec}")

    def decode_burst(self, frames: list[EncodedFrame]) -> np.ndarray:
        """Decode a batch of frames. Returns (N, H, W, C) uint8."""
        decoded = [self.decode_frame(f) for f in frames]
        return np.stack(decoded, axis=0)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize to model input dimensions."""
        h, w = self._config.target_height, self._config.target_width
        if image.shape[0] != h or image.shape[1] != w:
            image = self._resize(image, h, w)
        return image

    # ------------------------------------------------------------------
    # Decoders
    # ------------------------------------------------------------------

    def _decode_raw(self, frame: EncodedFrame) -> np.ndarray:
        arr = np.frombuffer(frame.data, dtype=np.uint8)
        return arr.reshape(frame.height, frame.width, frame.channels)

    def _decode_jpeg(self, frame: EncodedFrame) -> np.ndarray:
        if self._jpeg_decoder is not None:
            img = self._jpeg_decoder.decode(frame.data)
            return self.preprocess(img)

        return self._decode_pil(frame)

    def _decode_pil(self, frame: EncodedFrame) -> np.ndarray:
        from PIL import Image

        img = Image.open(io.BytesIO(frame.data)).convert("RGB")
        arr = np.array(img)
        return self.preprocess(arr)

    def _decode_video(self, frame: EncodedFrame) -> np.ndarray:
        """Decode a single video-encoded frame using PyAV."""
        try:
            import av

            container = av.open(io.BytesIO(frame.data))
            for vframe in container.decode(video=0):
                arr = vframe.to_ndarray(format="rgb24")
                return self.preprocess(arr)
        except ImportError:
            logger.warning("PyAV not available for video decoding")

        return np.zeros(
            (self._config.target_height, self._config.target_width, 3),
            dtype=np.uint8,
        )

    @staticmethod
    def _resize(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        try:
            import cv2

            return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image

            img = Image.fromarray(image)
            img = img.resize((target_w, target_h), Image.BILINEAR)
            return np.array(img)


class NvdecDecoder:
    """GPU-accelerated video decoding via NVIDIA NVDEC (optional)."""

    def __init__(self, gpu_id: int = 0) -> None:
        self._gpu_id = gpu_id
        self._available = self._check_available()

    @staticmethod
    def _check_available() -> bool:
        try:
            import PyNvCodec  # noqa: F401

            return True
        except ImportError:
            return False

    @property
    def available(self) -> bool:
        return self._available

    def decode_h264(self, data: bytes, width: int, height: int) -> np.ndarray:
        if not self._available:
            raise RuntimeError("NVDEC not available")

        import PyNvCodec as nvc

        nv_dec = nvc.PyNvDecoder(width, height, nvc.PixelFormat.NV12, nvc.CudaVideoCodec.H264, self._gpu_id)
        raw_surface = nv_dec.DecodeSingleSurface(data)
        if raw_surface.Empty():
            raise RuntimeError("NVDEC decode failed")

        to_rgb = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.NV12, nvc.PixelFormat.RGB, self._gpu_id)
        rgb_surface = to_rgb.Execute(raw_surface)

        downloader = nvc.PySurfaceDownloader(width, height, nvc.PixelFormat.RGB, self._gpu_id)
        frame_np = np.zeros((height, width, 3), dtype=np.uint8)
        downloader.DownloadSingleSurface(rgb_surface, frame_np)
        return frame_np
