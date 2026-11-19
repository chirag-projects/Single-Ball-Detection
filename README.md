# Real-Time Ball Detection & Trajectory Visualization

A Python computer vision application that detects a single sports ball through a webcam and displays its movement as a glowing 2D trajectory.

Built with **YOLOv8, OpenCV, and NumPy**.

## Features

- Live webcam-based sports ball detection.
- Pretrained YOLOv8 model without fixed ball-color filtering.
- Bounding box and center marker for the detected ball.
- Glowing trajectory with fading older segments.
- Missing-detection handling to avoid connecting paths across detection gaps.
- Keyboard controls to clear the trajectory and stop the application.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Ultralytics YOLOv8 | Sports ball detection |
| OpenCV | Webcam capture, drawing, and display |
| NumPy | Image-layer operations |
| `collections.deque` | Recent position history |

## How It Works

1. Capture a frame from the webcam.
2. Run YOLOv8 inference, filtering for the `sports ball` class.
3. Select at most one detection and calculate its bounding-box center.
4. Store the center in a rolling position history.
5. Draw trajectory segments with a blurred glow.
6. Display the result in a live camera window.

When no ball is detected, the application records a gap rather than drawing a line across the missing positions.

## Installation

### Requirements

- Python environment
- Webcam
- Internet access for package installation and the initial model download

### Install Dependencies

```bash
python -m pip install ultralytics opencv-python numpy
```

Save the application code as `live_tracker.py`.

## Run

Open a terminal in the project folder:

```bash
python live_tracker.py
```

The application loads `yolov8n.pt` and opens the webcam.

For an initial test, show one ball clearly against a simple background and move it slowly.

### Controls

| Key | Action |
|---|---|
| `C` | Clear the trajectory |
| `Q` | Quit the application |

Click the camera window before using keyboard controls.

## Configuration

Adjust these values in `live_tracker.py`:

| Setting | Default | Purpose |
|---|---|---|
| Camera index | `0` | Select the webcam |
| Model | `yolov8n.pt` | Pretrained detector |
| `imgsz` | `320` | Inference image size |
| `conf` | `0.25` | Detection confidence threshold |
| `max_det` | `1` | Maximum returned detections |
| `device` | `"cpu"` | Inference device |
| Trail length | `45` | Position history in processed frames |

Increasing image size may help detect smaller objects but can slow processing. Lowering the confidence threshold may introduce false detections.

## Limitations

- Detects objects recognized by the pretrained model as sports balls—not every possible ball.
- Shows an observed **2D image-space trajectory**, not a 3D path or future prediction.
- Uses frame-by-frame detection rather than persistent identity tracking.
- May switch between balls if multiple balls appear.
- Fast movement, blur, poor lighting, and occlusion can cause missed detections.
- Performance depends on the computer and camera.

## Troubleshooting

- **Webcam does not open:** Check camera permissions, close other camera applications, or try camera index `1`.
- **Ball is not detected:** Improve lighting, bring it closer, and test while holding it still.
- **Video is slow:** Check inference timing with `verbose=True` before changing settings.
- **Missing package error:** Install dependencies using the same Python environment used to run the script.

## Future Improvements

- Measure frame rate and detection accuracy.
- Add video recording and CSV trajectory export.
- Compare performance under motion blur and occlusion.
- Add motion prediction during short detection gaps.
- Train a custom detector for unsupported ball types.
