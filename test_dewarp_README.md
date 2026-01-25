# test_dewarp.py - Lens Distortion Correction Utility

Interactive utility for calibrating nvdewarper lens distortion correction on Jetson devices using DeepStream's nvdewarper plugin.

## Overview

This utility helps you visually calibrate lens distortion correction parameters for fisheye/wide-angle cameras. It uses NVIDIA's nvdewarper with `FISH_PERSPECTIVE` projection (type 4) to convert fisheye images to rectilinear (undistorted) images.

## Requirements

- NVIDIA Jetson device with DeepStream SDK 7.x installed
- CSI camera (IMX477 or similar)
- Python 3 with OpenCV (GStreamer support)

## Basic Usage

```bash
# Default: sensor 0, 1920x1080 @ 30fps
python3 test_dewarp.py

# Specify sensor ID
python3 test_dewarp.py --sensor-id 1

# Full 12MP resolution (IMX477)
python3 test_dewarp.py --sensor-id 1 --width 4032 --height 3040 --framerate 21

# With 180-degree rotation
python3 test_dewarp.py --sensor-id 1 --width 4032 --height 3040 --framerate 21 --flip-method 2

# Start with grid overlay
python3 test_dewarp.py --grid

# Start without dewarping (raw camera view)
python3 test_dewarp.py --no-dewarp
```

## Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--sensor-id` | 0 | CSI camera sensor ID |
| `--width` | 1920 | Capture width in pixels |
| `--height` | 1080 | Capture height in pixels |
| `--framerate` | 30 | Capture framerate (use 21 for 12MP) |
| `--flip-method` | 0 | Image rotation (0=none, 2=180°, see below) |
| `--no-dewarp` | false | Start with dewarping disabled |
| `--config` | auto | Path to config file (default: `dewarp_sensor{id}.txt`) |
| `--grid` | false | Show grid overlay on startup |

### Flip Method Values

- `0` - No rotation
- `1` - Counterclockwise 90°
- `2` - Rotate 180°
- `3` - Clockwise 90°
- `4` - Horizontal flip
- `6` - Vertical flip

## Interactive Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `d` | Toggle dewarp on/off (compare corrected vs raw) |
| `g` | Toggle grid overlay (helps see distortion) |
| `s` | Save current frame as PNG |
| `TAB` | Select next parameter to adjust |
| `+` / `UP` | Increase selected parameter |
| `-` / `DOWN` | Decrease selected parameter |
| `ENTER` | Apply changes (restarts camera with new config) |
| `r` | Reset all parameters to defaults |

## Parameters

### Distortion Coefficients (k1, k2, k3, k4)

These control the radial distortion correction:

- **k1** (step: 0.1) - Primary distortion coefficient
  - Negative values: correct barrel distortion (edges bent outward)
  - Positive values: correct pincushion distortion (edges bent inward)
  - Start with k1, typically between -0.5 and 0.5

- **k2, k3, k4** (step: 0.1) - Higher-order corrections
  - Usually smaller than k1
  - Adjust if k1 alone doesn't fully correct the distortion

### Focal Length (fx, fy)

Camera intrinsic parameters (step: 50):

- Controls the field of view of the corrected image
- Lower values = wider field of view (zoom out)
- Higher values = narrower field of view (zoom in)
- Default estimate: `width * 0.8` for fx, `height * 0.8` for fy

### Optical Center (cx, cy)

Principal point / image center (step: 10):

- Default: center of image (`width/2`, `height/2`)
- Adjust if the distortion center is offset from image center
- Usually close to the default values

## Calibration Workflow

1. **Start the utility with grid overlay:**
   ```bash
   python3 test_dewarp.py --sensor-id 1 --grid
   ```

2. **Point camera at straight lines** (door frame, building edge, ruler)

3. **Observe the distortion:**
   - Press `d` to toggle dewarp off and see raw distortion
   - Barrel distortion: straight lines bow outward
   - Pincushion distortion: straight lines bow inward

4. **Adjust k1 first:**
   - Press `TAB` until k1 is selected
   - Use `+`/`-` to adjust until lines appear straight
   - Press `ENTER` to apply

5. **Fine-tune with k2-k4 if needed:**
   - Adjust higher-order coefficients for remaining distortion
   - Apply after each adjustment

6. **Adjust focal length if image appears zoomed:**
   - Select fx/fy with `TAB`
   - Decrease for wider view, increase for narrower

7. **Save a test frame:**
   - Press `s` to save the corrected image
   - Verify quality before integrating into main system

## Configuration Files

The utility automatically creates and updates config files:

- Default location: `dewarp_sensor{id}.txt` (same directory as script)
- Format: DeepStream nvdewarper INI-style config

Example config file (`dewarp_sensor1.txt`):
```ini
[property]
output-width=4032
output-height=3040
num-batch-buffers=1

[surface0]
projection-type=4
surface-index=0
width=4032
height=3040
focal-length=3225.600000;2432.000000
distortion=-0.200000;0.100000;0.000000;0.000000
src-x0=2016.000000
src-y0=1520.000000
```

## Tips

- **Start with dewarp OFF** (`--no-dewarp`) to see raw distortion first
- **Use the grid overlay** (`g` key) to better visualize distortion
- **Make small adjustments** and apply frequently
- **k1 is usually enough** for most lenses - only use k2-k4 for stubborn distortion
- **Resolution matters** - calibrate at the resolution you'll actually use
- **The `*` indicator** shows unsaved changes - press ENTER to apply

## Troubleshooting

### "Invalid Projection type (0)"
- Config file is malformed or missing
- Delete the config file and restart - it will be recreated

### Camera fails to open
- Check sensor-id is correct
- Verify framerate is supported (21fps for 12MP on IMX477)
- Ensure no other process is using the camera

### Image appears cropped/wrong
- Delete old config file if resolution changed
- Ensure width/height in config matches command line arguments

## Integration

Once calibrated, use the same config file with the main vision system by integrating nvdewarper into the GStreamer pipeline. The config file format is compatible with DeepStream's nvdewarper element.
