# Sentiment Analysis of Cephalopods (GSoC 2026) — Entry Task

An open-source, multi-modal pipeline for automated cephalopod behavioral sentiment analysis.  
This project implements interpretable feature extraction, rule-based sentiment classification, and behavioral phase segmentation from video footage of cephalopods.

---

## ✅ Implemented Components

### 1. Dataset Ingestion Pipeline + Preprocessing (`src/data_loader.py`, `src/preprocess.py`)

A modular data ingestion pipeline handles video loading, validation, and preprocessing, ensuring consistent frame extraction across datasets.

```python
from src.data_loader import VideoLoader, BatchIngestor
from src.preprocess import Preprocessor, PreprocessConfig

# Single video
loader = VideoLoader("data/video_fixed.mp4")
print(loader.metadata)

# Batch ingestion
ingestor = BatchIngestor("data/")
ingestor.save_manifest("data/manifest.json")

# Preprocessing
config = PreprocessConfig(target_size=(224, 224), color_space="bgr", normalize=True)
preprocessor = Preprocessor(config)
frames = preprocessor.process_video(loader)
```

**Capabilities:**
- Video loading with format validation (`.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`)
- Metadata extraction (fps, resolution, codec, duration)
- Batch directory scanning with error reporting
- JSON manifest generation for dataset tracking
- Configurable resizing, color-space conversion, normalization
- FPS downsampling for compute-efficient processing
- Temporal windowing for sequence-based analysis

---

### 2. Behavioral Feature Extraction (`analyze_behavior.ipynb`)

Behavioral features include motion magnitude, color/pattern stability, and temporal segmentation into activity phases.

- **Optical Flow (Motion Magnitude):** Dense optical flow via `cv2.calcOpticalFlowFarneback` produces a time-series metric representing overall behavioral activity.
- **HSV Histograms (Color Stability):** Frame-to-frame color shift correlation maps directly to camouflage or stress-induced chromatophore states.
- **Heatmap Overlay:** Motion magnitude visualized as a color-mapped overlay on source footage (`results/output_heatmap.mp4`).

---

### 3. Multi-Modal Feature Baseline

A lightweight multi-modal baseline is established using interpretable visual features, with extensibility to integrate audio features in future iterations.

```python
features = {
    "motion": motion_magnitudes,      # Optical flow magnitude per frame
    "color":  color_stability_values,  # HSV histogram correlation per frame
}
```

---

### 4. Sentiment Label Classification (`src/classifier.py`)

A rule-based baseline classification system maps extracted behavioral signals to coarse sentiment states, providing an interpretable foundation for future learning-based approaches.

```python
from src.classifier import SentimentClassifier, BehavioralFeatures, format_report

features = BehavioralFeatures(
    motion_magnitude=motion_values,
    color_stability=color_values,
    fps=30.0,
)
classifier = SentimentClassifier()
result = classifier.classify(features)
print(format_report(result))
```

**Sentiment Taxonomy:**

| Label | Description | Feature Indicators |
|-------|-------------|-------------------|
| `CALM` | Resting / settled state | Low motion, moderate color stability |
| `ACTIVE` | Exploration or locomotion | Sustained high motion |
| `REACTIVE` | Startle / stimulus response | Sharp motion spike above baseline |
| `CAMOUFLAGED` | Active crypsis | Very low motion + high color stability |

**Output includes:**
- Per-frame label assignment with confidence scores
- Sentiment distribution summary
- Behavioral phase segmentation (contiguous labeled intervals)
- Adaptive thresholds derived from the data distribution

---

### 5. Evaluation & Signal Validation

Evaluation is performed using signal consistency metrics and qualitative alignment with observed behavior, with plans to incorporate labeled datasets in future work.

- **Activity ratio** (fraction of active vs. resting frames)
- **Phase count and duration** statistics
- **Threshold analysis** (adaptive percentile-based calibration)
- **Cross-feature consistency** (motion × color decoupling analysis)

---

### 6. Documentation + Reproducibility

- This README with full usage instructions
- Detailed behavioral analysis in [`ANALYSIS.md`](ANALYSIS.md)
- Reproducible Jupyter notebook ([`analyze_behavior.ipynb`](analyze_behavior.ipynb))
- Modular `src/` package with CLI entry points

---

## 🔧 System Roadmap

> This entry task establishes the foundational components of a broader multi-modal behavioral analysis system. The current implementation focuses on interpretable feature extraction and behavioral reasoning, which will be extended in future work to include classification models, multi-modal inputs, and deployment interfaces.

| Requirement | Current Implementation | Future Extension |
|-------------|----------------------|------------------|
| Dataset pipeline | Video loader + batch ingestion + preprocessing | Multi-species datasets, annotation tools |
| Multi-modal baseline | Motion + color features | Audio (hydrophone), pose estimation |
| Feature extractor | Optical flow + HSV histograms + phases | Contour analysis, localized chromatophore tracking |
| Classification | Rule-based label assignment | Learning-based classifiers (CNN/LSTM) |
| Evaluation | Signal metrics + behavioral validation | Labeled dataset benchmarks, F1/accuracy |
| Documentation | README + ANALYSIS.md + notebook | API docs, contributing guide |
| Deployment | *Planned* | REST API + interactive dashboard |

---

## Setup Instructions

### 1. Clone and enter the repository
```bash
git clone https://github.com/AR-droid/Sentiment-Analysis-of-Cephalopods.git
cd Sentiment-Analysis-of-Cephalopods
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
```

### 3. Install dependencies
```bash
pip install jupyter nbformat opencv-python matplotlib numpy
```

### 4. Run the analysis notebook
```bash
jupyter notebook analyze_behavior.ipynb
```

### 5. Run the classifier (CLI)
```bash
python src/classifier.py data/video_fixed.mp4
```

### 6. Run the data ingestion pipeline (CLI)
```bash
python src/data_loader.py data/          # Batch ingestion
python src/data_loader.py data/video_fixed.mp4  # Single video
```

---

## Project Structure

```
├── README.md               # This file
├── ANALYSIS.md             # Detailed behavioral analysis
├── analyze_behavior.ipynb  # Feature extraction + visualization notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Video ingestion pipeline
│   ├── preprocess.py       # Frame preprocessing
│   └── classifier.py       # Rule-based sentiment classification
├── data/
│   ├── video_fixed.mp4     # Input video
│   └── manifest.json       # (generated) Dataset manifest
├── results/
│   ├── feature_plots.png   # Feature visualization
│   └── output_heatmap.mp4  # Motion heatmap overlay
└── venv/                   # Virtual environment
```

---

## Short Analysis

The implemented optical flow feature accurately captures physical bursts of activity, representing movement behaviors like fleeing, aggressive striking, or explorative roaming. The HSV histogram feature provides a robust proxy for changes in chromatophore states (color/pattern changes), which are primary non-verbal indicators of stress or camouflage in most cephalopods. By utilizing moving average smoothing, we counteract high-frequency noise from water ripples or camera jitter. However, a key limitation is the dependency on fixed camera positions and clear lighting; dynamic backgrounds create severe noise for optical flow calculation. Integrating additional modalities, such as bioacoustic hydrophone data, could contextualize behavioral responses (e.g., verifying if the observed burst was an immediate flight response to a sound pulse), while posture-estimation keypoints would enhance semantic meaning beyond raw "movement vs stillness."
