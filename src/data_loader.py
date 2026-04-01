"""
Dataset Ingestion Pipeline
==========================
Modular video loading and metadata extraction for cephalopod behavioral
analysis. Supports single-file and batch ingestion with validation.

Usage:
    from src.data_loader import VideoLoader

    loader = VideoLoader("data/video_fixed.mp4")
    print(loader.metadata)
    frames = loader.load_frames()
"""

import cv2
import os
import json
import glob
from dataclasses import dataclass, asdict
from typing import List, Optional, Generator
import numpy as np


@dataclass
class VideoMetadata:
    """Structured metadata for an ingested video file."""
    filepath: str
    filename: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration_seconds: float
    codec: str
    is_valid: bool

    def to_dict(self) -> dict:
        return asdict(self)


class VideoLoader:
    """
    Loads and validates a single video file, exposing metadata and
    frame-level iteration.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Raises
    ------
    FileNotFoundError
        If the video file does not exist.
    ValueError
        If the file cannot be opened by OpenCV.
    """

    SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

    def __init__(self, video_path: str):
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        ext = os.path.splitext(video_path)[1].lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported format '{ext}'. "
                f"Supported: {self.SUPPORTED_EXTENSIONS}"
            )

        self._path = video_path
        self._metadata = self._extract_metadata()

    def _extract_metadata(self) -> VideoMetadata:
        """Open video and extract metadata without loading all frames."""
        cap = cv2.VideoCapture(self._path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self._path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])

        cap.release()

        return VideoMetadata(
            filepath=os.path.abspath(self._path),
            filename=os.path.basename(self._path),
            fps=round(fps, 2),
            width=width,
            height=height,
            total_frames=total_frames,
            duration_seconds=round(total_frames / fps, 2) if fps > 0 else 0.0,
            codec=codec,
            is_valid=total_frames > 0 and fps > 0,
        )

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def iter_frames(self) -> Generator[np.ndarray, None, None]:
        """
        Yield frames one at a time (memory-efficient).

        Yields
        ------
        np.ndarray
            BGR frame (H, W, 3).
        """
        cap = cv2.VideoCapture(self._path)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame
        finally:
            cap.release()

    def load_frames(self, max_frames: Optional[int] = None) -> List[np.ndarray]:
        """
        Load all frames into memory.

        Parameters
        ----------
        max_frames : int, optional
            Maximum number of frames to load. Loads all if None.

        Returns
        -------
        list of np.ndarray
            List of BGR frames.
        """
        frames = []
        for i, frame in enumerate(self.iter_frames()):
            if max_frames is not None and i >= max_frames:
                break
            frames.append(frame)
        return frames

    def __repr__(self) -> str:
        m = self._metadata
        return (
            f"VideoLoader('{m.filename}', {m.width}x{m.height}, "
            f"{m.fps}fps, {m.duration_seconds}s, {m.total_frames} frames)"
        )


class BatchIngestor:
    """
    Batch-load and validate all videos from a directory.

    Parameters
    ----------
    directory : str
        Path to directory containing video files.

    Usage:
        ingestor = BatchIngestor("data/raw/")
        for loader in ingestor.videos:
            print(loader.metadata)
        ingestor.save_manifest("data/manifest.json")
    """

    def __init__(self, directory: str):
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Not a directory: {directory}")

        self._directory = directory
        self._videos: List[VideoLoader] = []
        self._errors: List[str] = []
        self._ingest()

    def _ingest(self):
        """Scan directory and attempt to load every supported video."""
        patterns = [
            os.path.join(self._directory, f"*{ext}")
            for ext in VideoLoader.SUPPORTED_EXTENSIONS
        ]
        paths = sorted(
            path for pattern in patterns for path in glob.glob(pattern)
        )

        for path in paths:
            try:
                loader = VideoLoader(path)
                if loader.metadata.is_valid:
                    self._videos.append(loader)
                else:
                    self._errors.append(f"Invalid video (0 frames or 0 fps): {path}")
            except (ValueError, FileNotFoundError) as e:
                self._errors.append(f"{path}: {e}")

    @property
    def videos(self) -> List[VideoLoader]:
        return self._videos

    @property
    def errors(self) -> List[str]:
        return self._errors

    def save_manifest(self, output_path: str):
        """Write a JSON manifest of all successfully ingested videos."""
        manifest = {
            "total_videos": len(self._videos),
            "total_errors": len(self._errors),
            "videos": [v.metadata.to_dict() for v in self._videos],
            "errors": self._errors,
        }
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest saved: {output_path}")

    def __repr__(self) -> str:
        return (
            f"BatchIngestor('{self._directory}', "
            f"{len(self._videos)} videos, {len(self._errors)} errors)"
        )


# ── CLI usage ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "data/video_fixed.mp4"

    if os.path.isdir(target):
        ingestor = BatchIngestor(target)
        for v in ingestor.videos:
            m = v.metadata
            print(f"  {m.filename}: {m.width}x{m.height}, "
                  f"{m.fps}fps, {m.duration_seconds}s")
        if ingestor.errors:
            print(f"\n{len(ingestor.errors)} errors:")
            for e in ingestor.errors:
                print(f"  {e}")
        ingestor.save_manifest("data/manifest.json")
    else:
        loader = VideoLoader(target)
        print(loader)
        print(json.dumps(loader.metadata.to_dict(), indent=2))
