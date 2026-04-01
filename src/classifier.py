"""
Rule-Based Sentiment Classification System
===========================================
Maps extracted behavioral features to coarse sentiment labels using
interpretable, threshold-based rules. Designed as a strong baseline
that can be replaced by learned classifiers in future work.

Sentiment Taxonomy
------------------
    CALM        — Low motion, stable appearance. Resting or settled state.
    ACTIVE      — Sustained moderate-to-high motion. Exploration or locomotion.
    REACTIVE    — Sharp motion spike. Startle response or sudden stimulus.
    CAMOUFLAGED — Very low motion + high color stability. Active crypsis.

Usage:
    from src.classifier import SentimentClassifier, BehavioralFeatures

    features = BehavioralFeatures(
        motion_magnitude=motion_values,
        color_stability=color_values,
        fps=30.0,
    )
    classifier = SentimentClassifier()
    results = classifier.classify(features)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class SentimentLabel(str, Enum):
    """Cephalopod behavioral sentiment categories."""
    CALM = "calm"
    ACTIVE = "active"
    REACTIVE = "reactive"
    CAMOUFLAGED = "camouflaged"

    @property
    def color(self) -> Tuple[int, int, int]:
        """BGR color for each sentiment."""
        _colors = {
            "calm": (255, 150, 50),       # Soft Blue
            "active": (50, 50, 255),      # Bright Red
            "reactive": (0, 0, 180),      # Dark Red
            "camouflaged": (200, 180, 0)  # Teal/Cyan
        }
        return _colors[self.value]

    @property
    def description(self) -> str:
        _descriptions = {
            "calm": "Low activity — resting or settled state",
            "active": "Sustained motion — exploration or locomotion",
            "reactive": "Sharp motion spike — startle or stimulus response",
            "camouflaged": "Very low motion + high color stability — active crypsis",
        }
        return _descriptions[self.value]


@dataclass
class BehavioralFeatures:
    """
    Container for extracted behavioral feature time-series.

    Parameters
    ----------
    motion_magnitude : list or np.ndarray
        Per-frame mean optical flow magnitude.
    color_stability : list or np.ndarray
        Per-frame HSV histogram correlation (1.0 = identical to previous frame).
    fps : float
        Frames per second of the source video (for time conversion).
    """
    motion_magnitude: np.ndarray
    color_stability: np.ndarray
    fps: float

    def __post_init__(self):
        self.motion_magnitude = np.asarray(self.motion_magnitude, dtype=np.float64)
        self.color_stability = np.asarray(self.color_stability, dtype=np.float64)

        if len(self.motion_magnitude) != len(self.color_stability):
            raise ValueError(
                f"Feature lengths must match: motion={len(self.motion_magnitude)}, "
                f"color={len(self.color_stability)}"
            )

    @property
    def num_frames(self) -> int:
        return len(self.motion_magnitude)

    @property
    def duration(self) -> float:
        return self.num_frames / self.fps

    @property
    def time_axis(self) -> np.ndarray:
        return np.linspace(0, self.duration, self.num_frames)


@dataclass
class ClassificationResult:
    """Result of sentiment classification for a video."""
    per_frame_labels: List[SentimentLabel]
    per_frame_confidence: List[float]
    summary: Dict[str, float]          # label → percentage of total frames
    dominant_label: SentimentLabel
    phases: List[Dict]                  # list of {label, start_sec, end_sec, duration}
    thresholds_used: Dict[str, float]

    def to_dict(self) -> dict:
        return {
            "dominant_label": self.dominant_label.value,
            "summary": self.summary,
            "num_phases": len(self.phases),
            "phases": self.phases,
            "total_frames": len(self.per_frame_labels),
        }


class SentimentClassifier:
    """
    Rule-based behavioral sentiment classifier.

    Classifies each frame into a sentiment label based on motion magnitude
    and color stability thresholds, then segments into continuous behavioral
    phases.

    Parameters
    ----------
    motion_percentile_low : float
        Percentile of motion distribution below which frames are "low motion".
        Default: 30th percentile.
    motion_percentile_high : float
        Percentile above which frames are "high motion". Default: 85th.
    color_stability_threshold : float
        Histogram correlation above which color is considered "stable".
        Default: 0.995.
    spike_factor : float
        Factor above the rolling mean that triggers a "reactive" label.
        Default: 2.5x.
    min_phase_duration : float
        Minimum phase duration in seconds to be reported. Default: 0.3s.
    smoothing_kernel : int
        Moving-average kernel size for smoothing motion signal. Default: 5.
    """

    def __init__(
        self,
        motion_percentile_low: float = 30.0,
        motion_percentile_high: float = 85.0,
        color_stability_threshold: float = 0.995,
        spike_factor: float = 2.5,
        min_phase_duration: float = 0.3,
        smoothing_kernel: int = 5,
    ):
        self.motion_pct_low = motion_percentile_low
        self.motion_pct_high = motion_percentile_high
        self.color_threshold = color_stability_threshold
        self.spike_factor = spike_factor
        self.min_phase_duration = min_phase_duration
        self.smoothing_kernel = smoothing_kernel

    def _smooth(self, signal: np.ndarray) -> np.ndarray:
        """Apply simple moving-average smoothing."""
        kernel = np.ones(self.smoothing_kernel) / self.smoothing_kernel
        return np.convolve(signal, kernel, mode="same")

    def _detect_spikes(self, signal: np.ndarray) -> np.ndarray:
        """
        Detect frames where motion exceeds spike_factor × rolling mean.
        Returns boolean array.
        """
        rolling_mean = self._smooth(signal)
        global_mean = np.mean(signal)
        # spike = local value significantly exceeds both rolling and global baseline
        return signal > (self.spike_factor * np.maximum(rolling_mean, global_mean))

    def classify(self, features: BehavioralFeatures) -> ClassificationResult:
        """
        Classify each frame and segment into behavioral phases.

        Parameters
        ----------
        features : BehavioralFeatures
            Extracted feature time-series.

        Returns
        -------
        ClassificationResult
            Per-frame labels, phases, and summary statistics.
        """
        motion = features.motion_magnitude
        color = features.color_stability

        # Adaptive thresholds from the data distribution
        low_threshold = np.percentile(motion, self.motion_pct_low)
        high_threshold = np.percentile(motion, self.motion_pct_high)

        spikes = self._detect_spikes(motion)

        thresholds = {
            "motion_low": round(float(low_threshold), 4),
            "motion_high": round(float(high_threshold), 4),
            "color_stability": self.color_threshold,
            "spike_factor": self.spike_factor,
        }

        # ── Per-frame classification ─────────────────────────────
        labels: List[SentimentLabel] = []
        confidences: List[float] = []

        for i in range(features.num_frames):
            m = motion[i]
            c = color[i]
            is_spike = spikes[i]

            if is_spike:
                # Sharp spike → reactive (regardless of other features)
                labels.append(SentimentLabel.REACTIVE)
                confidences.append(min(1.0, m / high_threshold))

            elif m <= low_threshold and c >= self.color_threshold:
                # Low motion + very stable color → camouflaged
                labels.append(SentimentLabel.CAMOUFLAGED)
                confidences.append(float(c))

            elif m <= low_threshold:
                # Low motion, but some color change → calm (not crypsis)
                labels.append(SentimentLabel.CALM)
                confidences.append(1.0 - (m / low_threshold) if low_threshold > 0 else 1.0)

            elif m >= high_threshold:
                # Sustained high motion → active
                labels.append(SentimentLabel.ACTIVE)
                confidences.append(min(1.0, m / high_threshold))

            else:
                # Middle zone — lean towards calm
                labels.append(SentimentLabel.CALM)
                ratio = (m - low_threshold) / (high_threshold - low_threshold) if high_threshold > low_threshold else 0.5
                confidences.append(1.0 - ratio)

        # ── Summary statistics ────────────────────────────────────
        label_counts = {}
        for label in SentimentLabel:
            count = sum(1 for l in labels if l == label)
            label_counts[label.value] = round(count / len(labels) * 100, 1)

        dominant = max(label_counts, key=label_counts.get)

        # ── Phase segmentation ────────────────────────────────────
        phases = self._segment_phases(labels, features.fps, features.time_axis)

        return ClassificationResult(
            per_frame_labels=labels,
            per_frame_confidence=confidences,
            summary=label_counts,
            dominant_label=SentimentLabel(dominant),
            phases=phases,
            thresholds_used=thresholds,
        )

    def _segment_phases(
        self,
        labels: List[SentimentLabel],
        fps: float,
        time_axis: np.ndarray,
    ) -> List[Dict]:
        """
        Segment consecutive frames with the same label into phases.
        Filters out phases shorter than min_phase_duration.
        """
        if not labels:
            return []

        raw_phases = []
        current_label = labels[0]
        start_idx = 0

        for i in range(1, len(labels)):
            if labels[i] != current_label:
                raw_phases.append((current_label, start_idx, i - 1))
                current_label = labels[i]
                start_idx = i
        raw_phases.append((current_label, start_idx, len(labels) - 1))

        # Filter by minimum duration
        min_frames = int(self.min_phase_duration * fps)
        phases = []
        for label, start, end in raw_phases:
            duration_frames = end - start + 1
            if duration_frames >= min_frames:
                phases.append({
                    "label": label.value,
                    "start_sec": round(float(time_axis[start]), 2),
                    "end_sec": round(float(time_axis[min(end, len(time_axis) - 1)]), 2),
                    "duration_sec": round(duration_frames / fps, 2),
                    "num_frames": duration_frames,
                })

        return phases


def format_report(result: ClassificationResult) -> str:
    """
    Generate a human-readable classification report.

    Parameters
    ----------
    result : ClassificationResult
        Output of SentimentClassifier.classify().

    Returns
    -------
    str
        Formatted text report.
    """
    lines = [
        "=" * 60,
        "  CEPHALOPOD BEHAVIORAL SENTIMENT REPORT",
        "=" * 60,
        "",
        f"  Dominant State: {result.dominant_label.value.upper()}",
        f"  ({result.dominant_label.description})",
        "",
        "  ── Sentiment Distribution ──",
    ]

    for label, pct in sorted(result.summary.items(), key=lambda x: -x[1]):
        bar_len = int(pct / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        lines.append(f"    {label:<12s} {bar} {pct:5.1f}%")

    lines.append("")
    lines.append(f"  ── Behavioral Phases ({len(result.phases)} detected) ──")

    for i, phase in enumerate(result.phases):
        lines.append(
            f"    Phase {i+1}: {phase['label']:<12s} "
            f"{phase['start_sec']:5.1f}s → {phase['end_sec']:5.1f}s "
            f"({phase['duration_sec']:.1f}s)"
        )

    lines.append("")
    lines.append("  ── Thresholds Used ──")
    for k, v in result.thresholds_used.items():
        lines.append(f"    {k}: {v}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ── CLI usage ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from src.data_loader import VideoLoader
    import cv2

    video_path = sys.argv[1] if len(sys.argv) > 1 else "data/video_fixed.mp4"
    loader = VideoLoader(video_path)
    print(f"Loaded: {loader}")
    print()

    # ── Extract features (same approach as the notebook) ──
    print("Extracting features...")

    motion_magnitudes = []
    color_changes = []

    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(prev_hist, prev_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitudes.append(np.mean(mag))

        curr_hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(curr_hist, curr_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        color_changes.append(cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL))

        prev_gray = gray
        prev_hist = curr_hist

    cap.release()

    # ── Classify ──
    features = BehavioralFeatures(
        motion_magnitude=motion_magnitudes,
        color_stability=color_changes,
        fps=loader.metadata.fps,
    )

    classifier = SentimentClassifier()
    result = classifier.classify(features)
    print(format_report(result))
