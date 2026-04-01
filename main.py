import cv2
import numpy as np
import argparse
import os
from tqdm import tqdm
import json

from src.data_loader import VideoLoader
from src.classifier import SentimentClassifier, BehavioralFeatures, format_report
from src.visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Cephalopod Behavioral Sentiment Analysis Pipeline")
    parser.add_argument("--video", type=str, default="data/video_fixed.mp4", help="Path to input video")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    moments_dir = os.path.join(args.output_dir, "moments")
    os.makedirs(moments_dir, exist_ok=True)

    # 1. Load Video
    print(f"Loading video: {args.video}")
    loader = VideoLoader(args.video)
    metadata = loader.metadata
    print(f"  - {metadata.width}x{metadata.height} @ {metadata.fps}fps")

    # 2. Extract Features
    print("Extracting features (Optical Flow & HSV Correlations)...")
    motion_magnitudes = []
    color_stabilities = []
    frames = []

    cap = cv2.VideoCapture(args.video)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2HSV)
    prev_hist = cv2.calcHist([prev_hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(prev_hist, prev_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    
    # We store the first frame to keep things aligned
    frames.append(prev_frame)
    # For the first frame, motion is 0 and stability is 1.0
    motion_magnitudes.append(0.0)
    color_stabilities.append(1.0)

    pbar = tqdm(total=metadata.total_frames - 1)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Optical Flow
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_magnitudes.append(float(np.mean(mag)))

        # Color Stability
        curr_hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(curr_hist, curr_hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        color_stabilities.append(float(correlation))

        frames.append(frame)
        prev_gray = gray
        prev_hist = curr_hist
        pbar.update(1)
    
    cap.release()
    pbar.close()

    # 3. Classify Sentiment
    print("Classifying behavioral states...")
    features = BehavioralFeatures(
        motion_magnitude=np.array(motion_magnitudes),
        color_stability=np.array(color_stabilities),
        fps=metadata.fps
    )
    classifier = SentimentClassifier()
    result = classifier.classify(features)
    
    report_path = os.path.join(args.output_dir, "behavior_report.txt")
    report_text = format_report(result)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"Report saved to {report_path}")

    # 4. Generate Side-by-Side Video
    print("Generating side-by-side analysis video...")
    output_video_path = os.path.join(args.output_dir, "side_by_side_analysis.mp4")
    
    # Dimensions: (Width*2, Height + TimelineHeight)
    vis = Visualizer(metadata.width, metadata.height, metadata.fps)
    combined_width = metadata.width * 2
    combined_height = metadata.height 
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, metadata.fps, (combined_width, combined_height))

    # Recalculate flow for heatmap (or we could have stored it, but memory-wise this is safer)
    cap = cv2.VideoCapture(args.video)
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    for i in tqdm(range(len(frames))):
        frame = frames[i]
        
        # Heatmap for the right side
        if i == 0:
            heatmap = frame.copy()
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            heatmap = vis.generate_heatmap(frame, flow)
            prev_gray = gray
        
        # Combine
        sbs = vis.create_side_by_side(frame, heatmap)
        
        # Add Timeline
        sbs = vis.draw_timeline(sbs, result.per_frame_labels, i)
        
        out_video.write(sbs)

    cap.release()
    out_video.release()
    print(f"Analysis video saved to {output_video_path}")

    # 5. Extract Key Moments
    print("Saving key behavioral moments...")
    # Find top 3 motion peaks
    motion_arr = np.array(motion_magnitudes)
    # simple peak finding: sorted indices of motion
    top_indices = np.argsort(motion_arr)[-3:][::-1]
    
    for rank, idx in enumerate(top_indices):
        timestamp = idx / metadata.fps
        moment_path = os.path.join(moments_dir, f"moment_peak_{rank+1}_frame_{idx}.png")
        vis.save_key_moment(frames[idx], timestamp, moment_path)
        print(f"  - Saved peak {rank+1} at {timestamp:.2f}s")

    # 6. Generate Summary Plots
    print("Generating behavioral summary plots...")
    import matplotlib.pyplot as plt
    
    # 6a. Sentiment Distribution Plot
    labels = list(result.summary.keys())
    percentages = list(result.summary.values())
    
    # Manually mapping colors for simplicity in main.py
    color_map = {
        "calm": "#3296fa",       # Soft Blue
        "active": "#ff3232",      # Bright Red
        "reactive": "#b40000",    # Dark Red
        "camouflaged": "#00b4b4"  # Teal
    }
    plot_colors = [color_map.get(l, "#cccccc") for l in labels]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(labels, percentages, color=plot_colors)
    plt.xlabel("Percentage of Total Video Time (%)")
    plt.title("Behavioral Sentiment Distribution")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add labels on bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{width}%', va='center')

    summary_plot_path = os.path.join(args.output_dir, "sentiment_distribution.png")
    plt.tight_layout()
    plt.savefig(summary_plot_path)
    plt.close()
    print(f"Summary plot saved to {summary_plot_path}")

    print("\nUPGRADE COMPLETE: Pipeline finished successfully.")

if __name__ == "__main__":
    main()
