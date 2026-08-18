# Virtual Painter

> Draw in the air with your index finger — a webcam-based painting app built with MediaPipe hand tracking and OpenCV.

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-00A67E?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Virtual Painter turns your webcam into a canvas. It tracks your hand in real time, reads
which fingers are raised, and uses that to switch between two modes:

| Fingers raised | Mode | What happens |
| --- | --- | --- |
| Index only | **Draw** | A stroke follows your fingertip |
| Index + middle | **Select** | You can pick a colour or tool from the toolbar |

Strokes are drawn onto a separate canvas layer and composited back over the live video
using bitwise masking, so your drawing stays sharp instead of smearing with the camera feed.

## Features

- Real-time hand tracking at interactive frame rates, with an on-screen FPS counter
- Gesture-driven mode switching — no keyboard or mouse required
- Four brush colours (white, red, green, sky blue), an eraser, and a canvas reset button
- Toolbar selection by hovering the header strip with two fingers
- Adjustable brush (12 px) and eraser (50 px) thickness
- Persistent canvas layer that survives camera motion

## Requirements

- Python 3.8 or newer
- A working webcam

## Installation

```bash
git clone https://github.com/0bada11/virtual-painter.git
cd virtual-painter
pip install -r requirements.txt
```

## Usage

```bash
cd src
python virtual_painter.py
```

**Controls**

- Raise your **index finger** to draw.
- Raise your **index + middle fingers** to enter selection mode, then move to the toolbar
  at the top of the screen to pick a colour, the eraser, or reset.
- Press `Esc` to quit.

## How It Works

1. **Hand detection** — `hand_tracking.py` wraps MediaPipe Hands and returns 21 landmarks
   per hand as `[id, x, y]` pixel coordinates.
2. **Finger state** — `fingersUp()` compares each fingertip landmark against the joint two
   positions below it. Non-thumb fingers are checked on the y axis; the thumb is checked on
   the x axis, because it folds sideways rather than down.
3. **Mode selection** — the index/middle finger combination decides draw vs. select.
4. **Drawing** — strokes are written to `frameCanvas`, a black `numpy` array the same size
   as the frame.
5. **Compositing** — the canvas is thresholded to build an inverse mask. A bitwise AND
   punches a hole in the video frame where the drawing goes, then a bitwise OR fills that
   hole with the stroke colour.

## Project Structure

```
virtual-painter/
├── src/
│   ├── virtual_painter.py    # Main application loop
│   └── hand_tracking.py      # Reusable handDetector class (MediaPipe wrapper)
├── assets/
│   ├── header/               # Toolbar images, one per selected tool
│   │   └── slider/           # Brush-size slider graphics
│   └── header-psd/           # Photoshop sources for the toolbar art
├── requirements.txt
└── LICENSE
```

## Related Projects

Part of a series of MediaPipe hand-tracking projects:
[Gesture Volume Control](https://github.com/0bada11/gesture-volume-control) ·
[Finger Counter](https://github.com/0bada11/finger-counter) ·
[Hand Gesture Detection](https://github.com/0bada11/hand-gesture-detection)

## License

Released under the [MIT License](LICENSE).
