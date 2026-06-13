# Camera Calibration

A comprehensive Python toolkit for camera calibration using OpenCV. This project provides tools to calibrate your camera, remove lens distortion, and apply the calibration to live video feeds.

## Overview

Camera calibration is the process of estimating the parameters of a camera's lens and image sensor. These parameters can be used to correct for lens distortion, measure the size of an object in world units, or determine the location of the camera in the scene.

This toolkit includes:

1. **Image Capture Tool**: Capture calibration images from your camera
2. **Calibration Tool**: Process the calibration images to compute camera parameters
3. **Live Undistortion**: Apply the calibration to a live video feed

## Requirements

- Python 3.6+
- OpenCV 4.5+
- NumPy 1.20+
- Matplotlib 3.4+ (for visualization)

Install the required packages:

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Capture Calibration Images

You need multiple images of a chessboard pattern from different angles and positions. The script `capture_calibration_images.py` helps you capture these images:

#### USB Cameras

```bash
python capture_calibration_images.py
```
Controls:
- Press `c` to capture an image
- Press `q` or Escape to quit

The images will be saved in the `calibration_images` directory. Move them to `images/<CAMERA_TYPE_RESOLUTION>/` for calibration.

#### CSI Cameras

```bash
python test_dewarp.py --sensor-id 0 --width 1920 --height 1080 --framerate 60 --flip-method 2 # Camera 0
python test_dewarp.py --sensor-id 1 --width 4032 --height 3040 --framerate 21 --flip-method 2 # Camera 1
```
Controls:
- Press `s` to save frame as PNG
- Press `d` to toggle dewarp on/off
- Press `g` to toggle grid overlay
- Press `q` to quit

The images will be saved in the `calibration_images` directory. Move them to `images/<CAMERA_TYPE_RESOLUTION>/` for calibration.

### Step 2: Run Camera Calibration

Process the calibration images to compute the camera matrix and distortion coefficients:

```bash
python camera_calibration.py
```

You can specify a different camera type/resolution subfolder using the `CAMERA_TYPE_RESOLUTION` environment variable:

```bash
CAMERA_TYPE_RESOLUTION=4k_camera python camera_calibration.py
```

All paths are relative to the script's location. The calibration results will be saved in the `output/<CAMERA_TYPE_RESOLUTION>` directory:
- `calibration_data.pkl`: Complete calibration data in pickle format
- `camera_matrix.txt`: Camera matrix in text format
- `distortion_coefficients.txt`: Distortion coefficients in text format
- `nvdewarper_config.txt`: NVIDIA nvdewarper configuration file for lens correction
- Undistorted versions of the calibration images (if enabled)

### Step 3: Test the Calibration with Live Video

Apply the calibration to a live video feed:

#### USB Cameras

```bash
python live_undistortion.py
```

Controls:
- Press `d` to toggle between distorted and undistorted view
- Press `q` to quit

#### CSI Cameras

```bash
python test_dewarp.py --sensor-id 0 --width 1920 --height 1080 --framerate 60 --flip-method 2 # Camera 0
python test_dewarp.py --sensor-id 1 --width 4032 --height 3040 --framerate 21 --flip-method 2 # Camera 1
```

Controls:
- Press `q` to toggle dewarp on/off
- Press `g` to toggle grid overlay
- Press `q` to quit
- Press `TAB` to select next parameter (k1/k2/k3/k4/fx/fy/cx/cy)
- Press `UP/DOWN` or `+/-` to adjust selected parameter
- Press `ENTER` to apply changes (restarts camera)

If configuration file "dewarp_sensor<sensor-id>.txt" does not exist, it is created with default parmeters.  
Copy config for specific camera from `calibration_images/output/<CAMERA_TYPE_RESOLUTION>/code/nvdewarper_config.txt`.

## Configuration

All scripts use variables instead of command-line arguments for configuration. You can modify these variables at the top of each script:

### In `capture_calibration_images.py`:

```python
CAMERA_ID = 0  # Camera ID (usually 0 for built-in webcam)
CHESSBOARD_SIZE = (6, 8)  # Number of inner corners per chessboard row and column
OUTPUT_DIRECTORY = 'calibration_images'  # Directory to save calibration images
```

### In `camera_calibration.py`:

```python
CHESSBOARD_SIZE = (6, 8)  # Number of inner corners per chessboard row and column
SQUARE_SIZE = 1.7  # Size of a square in centimeters
SAVE_UNDISTORTED = True  # Whether to save undistorted images
USE_FISHEYE = True  # Use fisheye model (4 coefficients) vs standard model (5 coefficients)
```

Environment variable:
```bash
CAMERA_TYPE_RESOLUTION=new_camera  # Subfolder for images and output (default: "new_camera")
```

Paths are automatically set relative to the script location:
- Images: `<script_dir>/images/<CAMERA_TYPE_RESOLUTION>/*.png`
- Output: `<script_dir>/output/<CAMERA_TYPE_RESOLUTION>/`

### In `live_undistortion.py`:

```python
CAMERA_ID = 0  # Camera ID (usually 0 for built-in webcam)
CALIBRATION_FILE = 'output/calibration_data.pkl'  # Path to calibration data
```

## How It Works

### Camera Calibration Process

1. **Image Collection**: Capture multiple images of a chessboard pattern from different angles
2. **Corner Detection**: Detect the chessboard corners in each image
3. **Calibration**: Use the detected corners to compute the camera matrix and distortion coefficients
4. **Undistortion**: Apply the calibration to remove lens distortion from images

### Camera Model

The script supports two calibration models controlled by `USE_FISHEYE`:

**Fisheye Model** (`USE_FISHEYE = True`):
- Uses `cv2.fisheye.calibrate()`
- Produces 4 radial distortion coefficients (k1, k2, k3, k4)
- Compatible with NVIDIA nvdewarper FISH_PERSPECTIVE projection

**Standard Pinhole Model** (`USE_FISHEYE = False`):
- Uses `cv2.calibrateCamera()`
- Produces 5 distortion coefficients (k1, k2, p1, p2, k3)
- Standard pinhole model with radial and tangential distortion

Both models output:
- **Camera Matrix**: A 3x3 matrix containing the focal lengths and optical centers
- **Distortion Coefficients**: A vector containing the distortion coefficients

## Example Results

After calibration, you can expect:

1. **Undistorted Images**: Straight lines in the real world will appear straight in the images
2. **Accurate Measurements**: You can measure distances and sizes in the real world from the images
3. **3D Reconstruction**: You can use the calibration for 3D reconstruction or augmented reality applications

## Troubleshooting

### Common Issues

1. **Chessboard Not Detected**: Make sure the entire chessboard is visible in the image and well-lit
2. **Poor Calibration Results**: Use more images from different angles and positions
3. **Camera Not Found**: Check the CAMERA_ID parameter (usually 0 for built-in webcams)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV for providing the computer vision algorithms
- The OpenCV documentation for the camera calibration tutorial