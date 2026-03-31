"""
Video Preprocessing Pipeline
=============================
Frame-level preprocessing for downstream feature extraction and model input.
Handles resizing, color-space conversion, normalization, and temporal windowing.

Usage:
    from src.data_loader import VideoLoader
    from src.preprocess import Preprocessor

    loader = VideoLoader("data/video_fixed.mp4")
    preprocessor = Preprocessor(target_size=(224, 224))
    processed = preprocessor.process_video(loader)
"""

import cv2
import os
import numpy as np
from typing import List, Tuple, Optional, Generator
from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    """Configuration for the preprocessing pipeline."""
    target_size: Tuple[int, int] = (224, 224)   # (width, height)
    color_space: str = "bgr"                     # bgr | gray | hsv
    normalize: bool = True                       # scale pixels to [0, 1]
    sample_fps: Optional[float] = None           # downsample to this fps (None = keep original)


class Preprocessor:
    """
    Applies a configurable preprocessing pipeline to video frames.

    Parameters
    ----------
    config : PreprocessConfig, optional
        Preprocessing settings. Uses defaults if not provided.

    Supported color spaces: bgr, gray, hsv
    """

    _COLOR_CONVERTERS = {
        "bgr":  None,                        # no conversion (OpenCV native)
        "gray": cv2.COLOR_BGR2GRAY,
        "hsv":  cv2.COLOR_BGR2HSV,
    }

    def __init__(self, config: Optional[PreprocessConfig] = None):
        self.config = config or PreprocessConfig()

        if self.config.color_space not in self._COLOR_CONVERTERS:
            raise ValueError(
                f"Unsupported color space '{self.config.color_space}'. "
                f"Supported: {list(self._COLOR_CONVERTERS.keys())}"
            )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply the full preprocessing pipeline to a single frame.

        Parameters
        ----------
        frame : np.ndarray
            Raw BGR frame from OpenCV.

        Returns
        -------
        np.ndarray
            Preprocessed frame.
        """
        # 1. Resize
        processed = cv2.resize(
            frame,
            self.config.target_size,
            interpolation=cv2.INTER_AREA,
        )

        # 2. Color-space conversion
        converter = self._COLOR_CONVERTERS[self.config.color_space]
        if converter is not None:
            processed = cv2.cvtColor(processed, converter)

        # 3. Normalize to [0, 1]
        if self.config.normalize:
            processed = processed.astype(np.float32) / 255.0

        return processed

    def process_video(self, loader, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Process all frames from a VideoLoader instance.

        Parameters
        ----------
        loader : VideoLoader
            A loaded video source.
        max_frames : int, optional
            Limit number of frames to process.

        Returns
        -------
        list of np.ndarray
            Preprocessed frames.
        """
        frames = []
        fps = loader.metadata.fps
        sample_interval = 1  # process every frame by default

        if self.config.sample_fps and self.config.sample_fps < fps:
            sample_interval = int(round(fps / self.config.sample_fps))

        for i, frame in enumerate(loader.iter_frames()):
            if max_frames is not None and len(frames) >= max_frames:
                break
            if i % sample_interval != 0:
                continue
            frames.append(self.process_frame(frame))

        return frames

    def iter_processed(self, loader) -> Generator[np.ndarray, None, None]:
        """
        Memory-efficient generator that yields preprocessed frames one at a time.

        Parameters
        ----------
        loader : VideoLoader
            A loaded video source.

        Yields
        ------
        np.ndarray
            Preprocessed frame.
        """
        fps = loader.metadata.fps
        sample_interval = 1
        if self.config.sample_fps and self.config.sample_fps < fps:
            sample_interval = int(round(fps / self.config.sample_fps))

        for i, frame in enumerate(loader.iter_frames()):
            if i % sample_interval != 0:
                continue
            yield self.process_frame(frame)

    def create_temporal_windows(
        self,
        frames: List[np.ndarray],
        window_size: int = 16,
        stride: int = 8,
    ) -> List[np.ndarray]:
        """
        Split a sequence of frames into overlapping temporal windows.

        Parameters
        ----------
        frames : list of np.ndarray
            Preprocessed frames.
        window_size : int
            Number of frames per window.
        stride : int
            Step size between windows.

        Returns
        -------
        list of np.ndarray
            Each element is a (window_size, H, W, C) array.
        """
        windows = []
        for start in range(0, len(frames) - window_size + 1, stride):
            window = np.stack(frames[start : start + window_size], axis=0)
            windows.append(window)
        return windows

    def save_frames(self, frames: List[np.ndarray], output_dir: str):
        """
        Save preprocessed frames to disk as PNG images.

        Parameters
        ----------
        frames : list of np.ndarray
            Preprocessed frames (will be un-normalized for saving).
        output_dir : str
            Directory to write frames into.
        """
        os.makedirs(output_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            # Un-normalize for saving
            if self.config.normalize:
                save_frame = (frame * 255).clip(0, 255).astype(np.uint8)
            else:
                save_frame = frame

            # Convert back to BGR for saving if grayscale
            path = os.path.join(output_dir, f"frame_{i:05d}.png")
            cv2.imwrite(path, save_frame)

        print(f"✅ Saved {len(frames)} frames to {output_dir}/")


# ── CLI usage ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.data_loader import VideoLoader

    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/video_fixed.mp4"

    loader = VideoLoader(video_path)
    print(f"📹 Loaded: {loader}")

    config = PreprocessConfig(
        target_size=(224, 224),
        color_space="bgr",
        normalize=True,
        sample_fps=10.0,
    )
    preprocessor = Preprocessor(config)

    frames = preprocessor.process_video(loader)
    print(f"✅ Preprocessed {len(frames)} frames")
    print(f"   Shape: {frames[0].shape}")
    print(f"   Dtype: {frames[0].dtype}")
    print(f"   Value range: [{frames[0].min():.3f}, {frames[0].max():.3f}]")

    # Create temporal windows
    windows = preprocessor.create_temporal_windows(frames, window_size=16, stride=8)
    print(f"📦 Created {len(windows)} temporal windows of shape {windows[0].shape}")
