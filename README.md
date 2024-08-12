# Helmet Detection Video Script

This script leverages YOLOv8 for real-time helmet detection in video feeds. It identifies helmets within video frames and provides visual feedback by drawing bounding boxes around detected helmets.

## Features

- **Real-time Detection**: Detects helmets in real-time from video input.
- **Device Compatibility**: Automatically selects the GPU if available; otherwise, it falls back to the CPU.
- **Visual Feedback**: Draws bounding boxes around detected helmets with confidence scores.

## Setup Instructions

### 1. Prerequisites

Ensure that the following libraries are installed:

```bash
pip install opencv-python-headless torch ultralytics ttkthemes pillow
