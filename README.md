# Virtual Painter

A Python project that turns your webcam into an **interactive drawing board**.  
Using **MediaPipe hand tracking**, you can draw in the air with your fingers, select brush colors, erase, or reset the canvas.

---

## Features
- **Draw in the air** with your index finger.
- **Selection mode** (two fingers up):
  - Choose brush colors from the header bar.
  - Select eraser or reset the canvas.
- **Drawing mode** (one finger up):
  - Use different brush thicknesses.
  - Draw lines that follow your finger movements.
- **Canvas overlay** so your drawings stay on screen while the webcam feed continues.
- **FPS counter** to monitor performance.

---

## Requirements
Install Python 3.8+ and required libraries:

```bash
pip install opencv-python mediapipe numpy
