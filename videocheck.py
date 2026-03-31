import cv2

video_path = "data/video_fixed.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ Error: Cannot open video")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

duration = frame_count / fps

print("✅ Video Loaded Successfully")
print(f"FPS: {fps}")
print(f"Resolution: {width} x {height}")
print(f"Total Frames: {frame_count}")
print(f"Duration: {duration:.2f} seconds")

cap.release()