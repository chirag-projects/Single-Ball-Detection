import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch

device = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", device)
# Load your existing model.
model = YOLO("yolov8n.pt")

ball_class = next(
    class_id
    for class_id, name in model.names.items()
    if name == "sports ball"
)

# Recent ball positions. None marks a missing detection.
trail = deque(maxlen=45)
color = (255, 255, 0)  # Cyan glow; unrelated to physical ball color.

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Cannot open webcam. Check camera permissions.")

print("Starting camera. First prediction may take a moment.")
print("Press Q to quit or C to clear the trail.")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read a camera frame.")
            break

        frame = cv2.flip(frame, 1)

        result = model.predict(
            frame,
            classes=[ball_class],
            max_det=1,
            conf=0.25,
            imgsz=320,
            device=device,
            verbose=True
        )[0]

        center = None
        rectangle = None

        if result.boxes is not None and len(result.boxes) > 0:
            x1, y1, x2, y2 = (
                result.boxes.xyxy[0].cpu().numpy().astype(int)
            )

            center = (
                int((x1 + x2) / 2),
                int((y1 + y2) / 2)
            )
            rectangle = (x1, y1, x2, y2)

        trail.append(center)

        # Draw recent trajectory onto a separate layer.
        line_layer = np.zeros_like(frame)
        points = list(trail)

        for i in range(1, len(points)):
            start, end = points[i - 1], points[i]

            # Avoid connecting across missing detections.
            if start is None or end is None:
                continue

            brightness = 0.2 + 0.8 * i / max(1, len(points) - 1)
            segment_color = tuple(
                int(channel * brightness) for channel in color
            )

            cv2.line(
                line_layer, start, end,
                segment_color, 3, cv2.LINE_AA
            )

        # Soft glow plus bright central trail.
        glow = cv2.GaussianBlur(
            line_layer, (0, 0), sigmaX=6
        )
        display = cv2.addWeighted(frame, 1.0, glow, 1.5, 0)
        display = cv2.add(display, line_layer)

        if center is not None:
            x1, y1, x2, y2 = rectangle
            cv2.rectangle(
                display, (x1, y1), (x2, y2), color, 2
            )
            cv2.circle(display, center, 5, (255, 255, 255), -1)
            status = "Ball detected"
        else:
            status = "Ball not detected"

        cv2.putText(
            display, status, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (255, 255, 255), 2
        )

        cv2.imshow("Single Ball - Glowing Trail", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            trail.clear()

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()