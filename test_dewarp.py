#!/usr/bin/env python3
"""
Test utility for nvdewarper lens distortion correction on Jetson.

This script tests the nvdewarper GStreamer element to verify it works
with your cameras before integrating into the main vision system.

Usage:
    python3 test_dewarp.py                    # Test with default settings
    python3 test_dewarp.py --sensor-id 1      # Test hi-res camera (sensor 1)
    python3 test_dewarp.py --no-dewarp        # Compare without dewarping
    python3 test_dewarp.py --width 1920 --height 1080  # Custom resolution

Keys:
    q - Quit
    d - Toggle dewarp on/off (for comparison)
    s - Save current frame as PNG
    c - Create/update dewarp config file interactively
"""

import cv2
import argparse
import os
import sys
import time

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Camera type/resolution subfolder (e.g., "new_camera", "imx477_1920x1080")
# Can be set via environment variable CAMERA_TYPE_RESOLUTION
CAMERA_TYPE_RESOLUTION = os.environ.get('CAMERA_TYPE_RESOLUTION', 'new_camera')

# Directory to save captured images
IMAGES_DIR = os.path.join(SCRIPT_DIR, 'images', CAMERA_TYPE_RESOLUTION)


# Default dewarp configuration for a typical wide-angle lens
# These values will need calibration for your specific lenses
DEFAULT_DEWARP_CONFIG = """
# nvdewarper CSV configuration file
# Format: projection-type, top-angle, bottom-angle, pitch, yaw, roll, focal-length, width, height
#
# projection-type: 1=PushBroom, 2=VertRadCyl, 3=Cylindrical, 4=Equirectangular, 5=FishEye, 6=Panini
# Angles in degrees, focal-length as multiplier
#
# For barrel distortion correction (common in wide-angle lenses), start with:
# - projection-type: 5 (FishEye) or 3 (Cylindrical)
# - focal-length: 1.0 (adjust up to zoom in, down to zoom out)
# - angles: usually 0 unless camera is tilted
#
[surface0]
type=3
top-angle=0
bottom-angle=0
pitch=0
yaw=0
roll=0
focal-length=1.0
width={width}
height={height}
"""


def create_dewarp_config(width, height, config_path, focal_length=None, distortion=None,
                         center_x=None, center_y=None, dst_focal=None, projection_type=1):
    """Create a dewarp configuration file for nvdewarper.

    DeepStream nvdewarper projection types:
      1=PushBroom, 2=VertRadCyl, 3=Perspective_Perspective,
      4=FISH_PERSPECTIVE (fisheye to rectilinear - removes barrel distortion),
      5=FISH_FISH, 6=FISH_CYL, 7=FISH_EQUIRECT, 8=FISH_PANINI,
      9=PERSPECTIVE_EQUIRECT, 10=PERSPECTIVE_PANINI, etc.

    For lens distortion correction, use projection-type=1
    with calibration parameters from OpenCV's cv::fisheye::calibrate().

    Args:
        width, height: Output resolution
        config_path: Path to save config file
        focal_length: Tuple (fx, fy) from camera calibration, or single value
        distortion: Tuple (k1, k2, k3, k4) distortion coefficients
        center_x, center_y: Optical center (principal point), defaults to image center
        dst_focal: Output focal length tuple (fx, fy) for zoom effect, optional
        projection_type: nvdewarper projection type (default: 4=FISH_PERSPECTIVE)
    """
    # Default focal length estimate based on resolution (adjust based on your lens)
    if focal_length is None:
        # Rough estimate: focal_length ~= width for typical wide-angle lens
        fx = width * 0.8
        fy = height * 0.8
    elif isinstance(focal_length, (int, float)):
        fx = fy = float(focal_length)
    else:
        fx, fy = focal_length

    # Default to zero distortion (no correction) - needs calibration for real correction
    if distortion is None:
        k1, k2, k3, k4 = 0.0, 0.0, 0.0, 0.0
    else:
        k1, k2, k3, k4 = distortion

    # Default optical center is image center
    if center_x is None:
        center_x = width / 2.0
    if center_y is None:
        center_y = height / 2.0

    config_content = f"""# nvdewarper config for lens distortion correction
# Calibrate your camera with OpenCV to get accurate parameters

[property]
output-width={width}
output-height={height}
num-batch-buffers=1

[surface0]
projection-type={projection_type}
surface-index=0
width={width}
height={height}

# Camera intrinsic parameters from calibration
# focal-length: fx;fy from camera matrix
focal-length={fx:.6f};{fy:.6f}

# distortion: k1;k2;k3;k4 fisheye distortion coefficients
distortion={k1:.6f};{k2:.6f};{k3:.6f};{k4:.6f}

# Optical center (principal point) from camera matrix
src-x0={center_x:.6f}
src-y0={center_y:.6f}
"""

    # Add destination focal length if specified (for zoom effect)
    if dst_focal is not None:
        if isinstance(dst_focal, (int, float)):
            dfx = dfy = float(dst_focal)
        else:
            dfx, dfy = dst_focal
        config_content += f"\n# Destination focal length (lower = zoom out, higher = zoom in)\ndst-focal-length={dfx:.6f};{dfy:.6f}\n"

    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"Created dewarp config: {config_path}")
    print(f"  Projection type: {projection_type}")
    print(f"  Focal length: {fx:.1f}, {fy:.1f}")
    print(f"  Distortion: {k1:.4f}, {k2:.4f}, {k3:.4f}, {k4:.4f}")
    print(f"  Optical center: {center_x:.1f}, {center_y:.1f}")
    return config_path


def get_pipeline_with_dewarp(sensor_id, width, height, framerate, flip_method, config_file):
    """Build GStreamer pipeline with nvdewarper.

    nvdewarper requires RGBA format input, so we convert before dewarping.

    flip-method values:
      0 = none, 1 = counterclockwise 90, 2 = rotate 180, 3 = clockwise 90
      4 = horizontal flip, 5 = upper-right-diagonal, 6 = vertical flip, 7 = upper-left-diagonal
    """
    # IMPORTANT: queue before nvdewarper prevents CUDA/EGL race conditions
    # that cause "cuGraphicsSubResourceGetMappedArray failed" errors
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! video/x-raw(memory:NVMM), format=(string)RGBA ! "
        f"queue max-size-buffers=3 leaky=downstream ! "
        f"nvdewarper config-file={config_file} ! "
        f"queue max-size-buffers=3 leaky=downstream ! "
        f"nvvidconv ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1 sync=false"
    )


def get_pipeline_without_dewarp(sensor_id, width, height, framerate, flip_method):
    """Build GStreamer pipeline without nvdewarper (for comparison)."""
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
    )


def draw_grid(frame, grid_size=50, color=(0, 255, 0), thickness=1):
    """Draw a grid overlay to help visualize distortion."""
    h, w = frame.shape[:2]

    # Vertical lines
    for x in range(0, w, grid_size):
        cv2.line(frame, (x, 0), (x, h), color, thickness)

    # Horizontal lines
    for y in range(0, h, grid_size):
        cv2.line(frame, (0, y), (w, y), color, thickness)

    # Center crosshair
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - 30, cy), (cx + 30, cy), (0, 0, 255), 2)
    cv2.line(frame, (cx, cy - 30), (cx, cy + 30), (0, 0, 255), 2)

    return frame


def load_config(config_path, width, height):
    """Load existing config file and return parameters dict."""
    params = {
        'fx': width * 0.8,
        'fy': height * 0.8,
        'k1': 0.0,
        'k2': 0.0,
        'k3': 0.0,
        'k4': 0.0,
        'cx': width / 2.0,
        'cy': height / 2.0,
        'projection_type': 1,
    }

    if not os.path.exists(config_path):
        return params

    try:
        with open(config_path, 'r') as f:
            content = f.read()

        # Parse focal-length
        import re
        match = re.search(r'focal-length=([0-9.]+);([0-9.]+)', content)
        if match:
            params['fx'] = float(match.group(1))
            params['fy'] = float(match.group(2))

        # Parse distortion
        match = re.search(r'distortion=([0-9.-]+);([0-9.-]+);([0-9.-]+);([0-9.-]+)', content)
        if match:
            params['k1'] = float(match.group(1))
            params['k2'] = float(match.group(2))
            params['k3'] = float(match.group(3))
            params['k4'] = float(match.group(4))

        # Parse optical center
        match = re.search(r'src-x0=([0-9.]+)', content)
        if match:
            params['cx'] = float(match.group(1))
        match = re.search(r'src-y0=([0-9.]+)', content)
        if match:
            params['cy'] = float(match.group(1))

        # Parse projection type (use multiline search to exclude comments starting with #)
        match = re.search(r'^projection-type=(\d+)', content, re.MULTILINE)
        if match:
            params['projection_type'] = int(match.group(1))
            print(f"  Found projection-type in config: {match.group(1)}")
        else:
            print(f"  WARNING: projection-type not found in config, using default: {params['projection_type']}")

        print(f"Loaded config from {config_path}")
    except Exception as e:
        print(f"Warning: Could not parse config file: {e}")

    return params


def interactive_config(width, height, config_path):
    """Interactively adjust dewarp parameters."""
    print("\n=== Interactive Dewarp Configuration ===")
    print("For accurate distortion correction, you need to calibrate your camera")
    print("using OpenCV's fisheye calibration with a checkerboard pattern.")
    print("\nFor now, you can estimate parameters:")

    try:
        print(f"\nFocal length estimate (typical: {width*0.6:.0f}-{width*1.2:.0f} for this resolution)")
        fx = float(input(f"  Focal length X [{width*0.8:.0f}]: ") or str(width*0.8))
        fy = float(input(f"  Focal length Y [{height*0.8:.0f}]: ") or str(height*0.8))

        print("\nDistortion coefficients (start with small values like -0.1 to 0.1)")
        print("  Negative k1 = pincushion correction, Positive k1 = barrel correction")
        k1 = float(input("  k1 [0.0]: ") or "0.0")
        k2 = float(input("  k2 [0.0]: ") or "0.0")
        k3 = float(input("  k3 [0.0]: ") or "0.0")
        k4 = float(input("  k4 [0.0]: ") or "0.0")

        print(f"\nOptical center (image center = {width/2:.0f}, {height/2:.0f})")
        cx = float(input(f"  Center X [{width/2:.0f}]: ") or str(width/2))
        cy = float(input(f"  Center Y [{height/2:.0f}]: ") or str(height/2))

    except ValueError:
        print("Invalid input, using defaults")
        fx, fy = width * 0.8, height * 0.8
        k1, k2, k3, k4 = 0.0, 0.0, 0.0, 0.0
        cx, cy = width / 2, height / 2

    create_dewarp_config(width, height, config_path,
                        focal_length=(fx, fy),
                        distortion=(k1, k2, k3, k4),
                        center_x=cx, center_y=cy)
    return config_path


def main():
    parser = argparse.ArgumentParser(description='Test nvdewarper lens distortion correction')
    parser.add_argument('--sensor-id', type=int, default=0, help='Camera sensor ID (default: 0)')
    parser.add_argument('--width', type=int, default=1920, help='Capture width (default: 1920)')
    parser.add_argument('--height', type=int, default=1080, help='Capture height (default: 1080)')
    parser.add_argument('--framerate', type=int, default=30, help='Framerate (default: 30)')
    parser.add_argument('--flip-method', type=int, default=0, help='Flip method (default: 0)')
    parser.add_argument('--no-dewarp', action='store_true', help='Start without dewarping')
    parser.add_argument('--config', type=str, default=None, help='Path to dewarp config file')
    parser.add_argument('--grid', action='store_true', help='Show grid overlay')
    args = parser.parse_args()

    # Determine config file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(script_dir, f'dewarp_sensor{args.sensor_id}.txt')

    # Load existing config or use defaults
    params = load_config(config_path, args.width, args.height)
    print(f"DEBUG: After load_config, projection_type = {params['projection_type']}")

    # Which parameter is currently selected for adjustment
    param_names = ['k1', 'k2', 'k3', 'k4', 'fx', 'fy', 'cx', 'cy']
    current_param_idx = 0  # Start with k1

    # Create initial config if it doesn't exist
    if not os.path.exists(config_path):
        print(f"Creating initial dewarp config: {config_path}")
        create_dewarp_config(args.width, args.height, config_path)

    dewarp_enabled = not args.no_dewarp
    show_grid = args.grid
    cap = None

    def update_config_and_reopen():
        """Update config file and reopen camera with new settings."""
        nonlocal cap
        create_dewarp_config(
            args.width, args.height, config_path,
            focal_length=(params['fx'], params['fy']),
            distortion=(params['k1'], params['k2'], params['k3'], params['k4']),
            center_x=params['cx'], center_y=params['cy'],
            projection_type=params['projection_type']
        )
        if dewarp_enabled:
            cap.release()
            cap = open_camera(True)

    def open_camera(with_dewarp):
        """Open camera with or without dewarp."""
        if with_dewarp:
            pipeline = get_pipeline_with_dewarp(
                args.sensor_id, args.width, args.height,
                args.framerate, args.flip_method, config_path
            )
            print(f"\nOpening camera WITH dewarp:")
        else:
            pipeline = get_pipeline_without_dewarp(
                args.sensor_id, args.width, args.height,
                args.framerate, args.flip_method
            )
            print(f"\nOpening camera WITHOUT dewarp:")

        print(f"  Pipeline: {pipeline[:80]}...")
        return cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    # Initial camera open
    cap = open_camera(dewarp_enabled)

    if not cap.isOpened():
        print("\nERROR: Failed to open camera!")
        if dewarp_enabled:
            print("nvdewarper may not be available or config file is invalid.")
            print("Try running with --no-dewarp to test without dewarping.")
            print(f"\nConfig file: {config_path}")
            if os.path.exists(config_path):
                print("Config contents:")
                with open(config_path, 'r') as f:
                    print(f.read())
        sys.exit(1)

    print(f"\nCamera opened successfully!")
    print(f"Resolution: {args.width}x{args.height} @ {args.framerate}fps")
    print(f"Dewarp: {'ENABLED' if dewarp_enabled else 'DISABLED'}")
    print(f"Images will be saved to: {IMAGES_DIR}")
    print(f"\nControls:")
    print("  q - Quit")
    print("  d - Toggle dewarp on/off")
    print("  g - Toggle grid overlay")
    print("  s - Save frame as PNG to images folder")
    print("  TAB - Select next parameter (k1/k2/k3/k4/fx/fy/cx/cy)")
    print("  UP/DOWN or +/- - Adjust selected parameter")
    print("  ENTER - Apply changes (restarts camera)")
    print("  r - Reset all parameters to defaults")

    # For FPS calculation
    fps_start = time.time()
    fps_count = 0
    fps = 0.0
    params_changed = False  # Track if params changed since last apply

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            time.sleep(0.1)
            continue

        # FPS calculation
        fps_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps = fps_count / elapsed
            fps_count = 0
            fps_start = time.time()

        # Draw grid if enabled
        display_frame = frame.copy()
        if show_grid:
            display_frame = draw_grid(display_frame)

        # Resize for display if too large (do this BEFORE adding text)
        display_h, display_w = display_frame.shape[:2]
        max_display = 1280
        if display_w > max_display:
            scale = max_display / display_w
            display_frame = cv2.resize(display_frame,
                                       (int(display_w * scale), int(display_h * scale)))

        # Add status text (after resize so text is readable)
        status = f"Dewarp: {'ON' if dewarp_enabled else 'OFF'} | Type: {params['projection_type']} | FPS: {fps:.1f}"
        cv2.putText(display_frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show current parameter being adjusted
        current_param = param_names[current_param_idx]
        param_value = params[current_param]

        # Format display based on parameter type
        if current_param in ['k1', 'k2', 'k3', 'k4']:
            param_str = f"{current_param}={param_value:+.3f}"
            step_str = "step=0.05"
        elif current_param in ['fx', 'fy']:
            param_str = f"{current_param}={param_value:.1f}"
            step_str = "step=50"
        else:  # cx, cy
            param_str = f"{current_param}={param_value:.1f}"
            step_str = "step=10"

        # Show all distortion params
        dist_str = f"k1={params['k1']:+.3f} k2={params['k2']:+.3f} k3={params['k3']:+.3f} k4={params['k4']:+.3f}"
        focal_str = f"fx={params['fx']:.0f} fy={params['fy']:.0f} cx={params['cx']:.0f} cy={params['cy']:.0f}"

        changed_indicator = " *" if params_changed else ""
        cv2.putText(display_frame, f"Adjusting: [{param_str}] ({step_str}) TAB=next, +/-=adjust, ENTER=apply{changed_indicator}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(display_frame, dist_str,
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(display_frame, focal_str,
                   (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow('Dewarp Test', display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord('d'):
            # Toggle dewarp
            dewarp_enabled = not dewarp_enabled
            cap.release()
            cap = open_camera(dewarp_enabled)
            if not cap.isOpened():
                print("Failed to reopen camera, reverting...")
                dewarp_enabled = not dewarp_enabled
                cap = open_camera(dewarp_enabled)

        elif key == ord('g'):
            show_grid = not show_grid
            print(f"Grid: {'ON' if show_grid else 'OFF'}")

        elif key == ord('s'):
            # Save frame to images/<CAMERA_TYPE_RESOLUTION>/
            if not os.path.exists(IMAGES_DIR):
                os.makedirs(IMAGES_DIR)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(IMAGES_DIR, f"dewarp_test_{timestamp}.jpg")
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"Saved: {filename}")

        elif key == 9:  # TAB
            # Select next parameter
            current_param_idx = (current_param_idx + 1) % len(param_names)
            print(f"Selected parameter: {param_names[current_param_idx]}")

        elif key == ord('+') or key == ord('=') or key == 82:  # + or UP arrow
            # Increase current parameter
            current_param = param_names[current_param_idx]
            if current_param in ['k1', 'k2', 'k3', 'k4']:
                params[current_param] += 0.05
            elif current_param in ['fx', 'fy']:
                params[current_param] += 50
            else:  # cx, cy
                params[current_param] += 10
            params_changed = True
            print(f"{current_param} = {params[current_param]:.3f}")

        elif key == ord('-') or key == 84:  # - or DOWN arrow
            # Decrease current parameter
            current_param = param_names[current_param_idx]
            if current_param in ['k1', 'k2', 'k3', 'k4']:
                params[current_param] -= 0.05
            elif current_param in ['fx', 'fy']:
                params[current_param] -= 50
            else:  # cx, cy
                params[current_param] -= 10
            params_changed = True
            print(f"{current_param} = {params[current_param]:.3f}")

        elif key == 13 or key == 10:  # ENTER
            # Apply changes
            if params_changed:
                print("Applying changes...")
                update_config_and_reopen()
                params_changed = False
            else:
                print("No changes to apply")

        elif key == ord('r'):
            # Reset to defaults
            params['fx'] = args.width * 0.8
            params['fy'] = args.height * 0.8
            params['k1'] = 0.0
            params['k2'] = 0.0
            params['k3'] = 0.0
            params['k4'] = 0.0
            params['cx'] = args.width / 2.0
            params['cy'] = args.height / 2.0
            params_changed = True
            print("Parameters reset to defaults. Press ENTER to apply.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nTest complete.")
    print(f"Dewarp config saved at: {config_path}")
    print(f"Images saved to: {IMAGES_DIR}")


if __name__ == "__main__":
    main()
