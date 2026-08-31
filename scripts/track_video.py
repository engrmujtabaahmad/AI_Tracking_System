from ultralytics import YOLO
import cv2

print("=" * 50)
print("Loading YOLO11 Model...")
print("=" * 50)

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open video
video_path = "assets/Traffic1.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Cannot open video -> {video_path}")
    exit()

frame_number = 0

while True:
    success, frame = cap.read()

    if not success:
        print("\nVideo Finished.")
        break

    frame_number += 1

    # Perform tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False
    )

    result = results[0]

    print(f"\n========== Frame {frame_number} ==========")

    # Check detections
    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes

        # Check if tracker generated IDs
        if boxes.id is not None:

            ids = boxes.id.cpu().numpy().astype(int)
            classes = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()

            for track_id, cls, conf in zip(ids, classes, confidences):

                class_name = model.names[cls]

                print(
                    f"ID: {track_id:3d} | "
                    f"Object: {class_name:10s} | "
                    f"Confidence: {conf:.2f}"
                )

        else:
            print("No Track IDs Found!")

    else:
        print("No Objects Detected!")

    # Draw results
    annotated_frame = result.plot()

    cv2.imshow("YOLO11 + ByteTrack Tracking", annotated_frame)

    key = cv2.waitKey(25)

    if key == ord("q"):
        print("Stopped by User.")
        break

cap.release()
cv2.destroyAllWindows()

print("\nTracking Completed Successfully!")