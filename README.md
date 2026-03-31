# Sentiment Analysis of Cephalopods (GSoC 2026) - Entry Task

This repository contains an open-source multi-modal pipeline for automated cephalopod behavioral sentiment analysis, fulfilling the GSoC 2026 Entry Task requirements. 

## Features
- **Data Handling**: Automatically loads `data/video_fixed.mp4`, extracting resolution, framerate, and duration metadata.
- **Feature Extraction**:
    - **Optical Flow (Motion Magnitude)**: Computes dense optical flow (`cv2.calcOpticalFlowFarneback`) to create a time-series metric representing overall behavioral activity levels.
    - **HSV Histograms (Color/Pattern Stability)**: Analyzes frame-to-frame color shifts using correlation, directly mapping to camouflage or stress-induced states.
- **Visualization**:
    - **Heatmap Video Overlay**: Generates an MP4 (`results/output_heatmap.mp4`) that overlays the motion magnitude directly onto the source footage.
    - **Feature Plots**: Utilizes Matplotlib to present smoothed feature metrics plotted against the video timeline (`results/feature_plots.png`).

## Setup Instructions

1. **Clone the repository** (if applicable) and enter the directory.
2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Or `venv\Scripts\activate` on Windows
   ```
3. **Install Dependencies**:
   ```bash
   pip install jupyter nbformat opencv-python matplotlib numpy
   ```
4. **Run the Analysis**:
   Launch a Jupyter Notebook server:
   ```bash
   jupyter notebook analyze_behavior.ipynb
   ```
   Or execute heavily from the command line:
   ```bash
   jupyter nbconvert --execute --to notebook --inplace analyze_behavior.ipynb
   ```

## Short Analysis
The implemented optical flow feature accurately captures physical bursts of activity, representing movement behaviors like fleeing, aggressive striking, or explorative roaming. The HSV histogram feature provides a robust proxy for changes in chromatophore states (color/pattern changes), which are primary non-verbal indicators of stress or camouflage in most cephalopods. By utilizing moving average smoothing, we counteract high-frequency noise from water ripples or camera jitter. However, a key limitation is the dependency on fixed camera positions and clear lighting; dynamic backgrounds create severe noise for optical flow calculation. Integrating additional modalities, such as bioacoustic hydrophone data, could contextualize behavioral responses (e.g., verifying if the observed burst was an immediate flight response to a sound pulse), while posture-estimation keypoints would enhance semantic meaning beyond raw "movement vs stillness."
