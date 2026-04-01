"""
Cephalopod Analysis Visualizer
===============================
Dedicated functions for generating high-impact video results, including:
- Side-by-side (Original vs. Heatmap)
- Behavioral Timeline Overlays
- Frame Annotations
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from src.classifier import SentimentLabel, ClassificationResult

class Visualizer:
    """
    Handles the generation of annotated video outputs and frame visualizations.
    """

    def __init__(self, width: int, height: int, fps: float):
        self.width = width
        self.height = height
        self.fps = fps
        self.timeline_height = 40

    def generate_heatmap(self, frame: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """
        Create a motion heatmap overlay on the frame.
        """
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        mag_norm = mag_norm.astype(np.uint8)

        # Apply a colormap
        heatmap = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)

        # Blend with original frame
        alpha = 0.6
        overlay = cv2.addWeighted(frame, 1 - alpha, heatmap, alpha, 0)
        return overlay

    def create_side_by_side(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """
        Combine two frames side-by-side.
        """
        # Ensure they have the same height
        h_l, w_l = left.shape[:2]
        h_r, w_r = right.shape[:2]

        if h_l != h_r:
            right = cv2.resize(right, (int(w_r * h_l / h_r), h_l))

        combined = np.hstack((left, right))
        return combined

    def draw_timeline(
        self,
        canvas: np.ndarray,
        labels: List[SentimentLabel],
        current_idx: int,
    ) -> np.ndarray:
        """
        Draw a behavioral timeline at the bottom of the frame.
        Active states (Red) and Calm states (Blue) are mapped to the bottom bar.
        """
        h, w = canvas.shape[:2]
        bar_h = self.timeline_height
        margin = 10

        # Create a blank bar at the bottom
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (30, 30, 30), -1)

        # Draw segments
        num_frames = len(labels)
        segment_w = w / num_frames

        for i, label in enumerate(labels):
            x1 = int(i * segment_w)
            x2 = int((i + 1) * segment_w)
            color = label.color
            cv2.rectangle(overlay, (x1, h - bar_h + 5), (x2, h - 5), color, -1)

        # Draw current position cursor
        cursor_x = int(current_idx * segment_w)
        cv2.line(overlay, (cursor_x, h - bar_h), (cursor_x, h), (255, 255, 255), 2)

        # Blend
        cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)

        # Add Legend/Text labels
        current_label = labels[current_idx]
        cv2.putText(
            canvas,
            f"STATE: {current_label.value.upper()}",
            (20, h - bar_h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            current_label.color,
            2,
        )

        return canvas

    def save_key_moment(self, frame: np.ndarray, timestamp: float, output_path: str):
        """
        Annotate and save a frame as a 'key moment'.
        """
        annotated = frame.copy()
        text = f"Key Moment: {timestamp:.2f}s"
        cv2.putText(
            annotated,
            text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3,
        )
        cv2.imwrite(output_path, annotated)
