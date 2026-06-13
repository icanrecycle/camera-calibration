import numpy as np
import cv2
import glob
import os
import pickle
import threading

CORNER_DETECTION_TIMEOUT = 30  # seconds per image

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Camera type/resolution subfolder (e.g., "new_camera", "4k_camera")
# Can be set via environment variable CAMERA_TYPE_RESOLUTION
CAMERA_TYPE_RESOLUTION = os.environ.get('CAMERA_TYPE_RESOLUTION', 'new_camera')

# Camera calibration parameters
# You can modify these variables as needed
# https://calib.io/pages/camera-calibration-pattern-generator
CHESSBOARD_SIZE = (6, 8)  # Number of inner corners per chessboard row and column
SQUARE_SIZE = 1.7         # Size of a square in centimeters
CALIBRATION_IMAGES_DIR = os.path.join(SCRIPT_DIR, f'images/{CAMERA_TYPE_RESOLUTION}')  # Path to calibration images
OUTPUT_DIRECTORY = os.path.join(SCRIPT_DIR, f'output/{CAMERA_TYPE_RESOLUTION}')  # Directory to save calibration results
SAVE_UNDISTORTED = True   # Whether to save undistorted images

# Fisheye calibration mode
# True: Use cv2.fisheye.calibrate() - produces 4 radial coefficients (k1,k2,k3,k4)
#       Compatible with nvdewarper FISH_PERSPECTIVE projection
# False: Use cv2.calibrateCamera() - produces 5 coefficients (k1,k2,p1,p2,k3)
#        Standard pinhole model with tangential distortion
USE_FISHEYE = True

def calibrate_camera():
    """
    Calibrate the camera using chessboard images.
    
    Returns:
        ret: The RMS re-projection error
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    
    # Scale object points by square size (for real-world measurements)
    objp = objp * SQUARE_SIZE
    
    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images (png + jpg)
    images = glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, '*.png')) + \
             glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, '*.jpg')) + \
             glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, '*.jpeg'))
    
    if not images:
        print(f"No calibration images found in {CALIBRATION_IMAGES_DIR}")
        return None, None, None, None, None
    
    # Create output directory if it doesn't exist
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
    
    print(f"Found {len(images)} calibration images")
    
    # Process each calibration image
    for idx, fname in enumerate(images):
        print(f"Processing image {idx+1}/{len(images)}: {fname}...", end=' ', flush=True)
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

         # Find the chessboard corners (run in daemon thread so it can be skipped/interrupted)
        result = [None]
        def detect_corners():
            result[0] = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        t = threading.Thread(target=detect_corners, daemon=True)
        t.start()
        t.join(timeout=CORNER_DETECTION_TIMEOUT)
        if t.is_alive():
            print(f"TIMEOUT after {CORNER_DETECTION_TIMEOUT}s - skipped")
            continue
        ret, corners = result[0]

        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
            
            # Save image with corners drawn
            output_img_path = os.path.join(OUTPUT_DIRECTORY, f'corners_{os.path.basename(fname)}')
            cv2.imwrite(output_img_path, img)
            
            print("Chessboard found")
        else:
            print("Chessboard NOT found")
    
    if not objpoints:
        print("No chessboard patterns were detected in any images.")
        return None, None, None, None, None
    
    print("Calibrating camera...")

    if USE_FISHEYE:
        # Fisheye calibration - produces 4 radial distortion coefficients (k1,k2,k3,k4)
        # Fisheye requires Nx1x3 object points and Nx1x2 image points
        objpoints_fish = [op.reshape(-1, 1, 3).astype(np.float32) for op in objpoints]
        imgpoints_fish = [ip.reshape(-1, 1, 2).astype(np.float32) for ip in imgpoints]

        K = np.zeros((3, 3))
        D = np.zeros((4, 1))

        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_FIX_SKEW
        )

        ret, mtx, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints_fish, imgpoints_fish, gray.shape[::-1],
            K, D, flags=calibration_flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
        print("Using FISHEYE calibration model (4 radial coefficients)")
    else:
        # Standard pinhole calibration - produces 5 distortion coefficients
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        print("Using STANDARD calibration model (5 coefficients)")

    # Save calibration results
    calibration_data = {
        'camera_matrix': mtx,
        'distortion_coefficients': dist,
        'rotation_vectors': rvecs,
        'translation_vectors': tvecs,
        'reprojection_error': ret,
        'fisheye_model': USE_FISHEYE
    }

    with open(os.path.join(OUTPUT_DIRECTORY, 'calibration_data.pkl'), 'wb') as f:
        pickle.dump(calibration_data, f)

    # Save camera matrix and distortion coefficients as text files
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'camera_matrix.txt'), mtx)
    np.savetxt(os.path.join(OUTPUT_DIRECTORY, 'distortion_coefficients.txt'), dist)

    print(f"Calibration complete! RMS re-projection error: {ret}")
    print(f"Results saved to {OUTPUT_DIRECTORY}")

    # Generate and save nvdewarper config file
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    img_h, img_w = gray.shape[:2]

    if USE_FISHEYE:
        k1, k2, k3, k4 = dist.flatten()
        dist_str = f"{k1:.6f};{k2:.6f};{k3:.6f};{k4:.6f}"
    else:
        coeffs = dist.flatten()
        dist_str = f"{coeffs[0]:.6f};{coeffs[1]:.6f};{coeffs[4]:.6f};0.0"

    nvdewarper_config = f"""# nvdewarper config for lens distortion correction
# Generated by camera_calibration.py
# Model: {'FISHEYE' if USE_FISHEYE else 'STANDARD'} (RMS error: {ret:.6f})

[property]
output-width={img_w}
output-height={img_h}
num-batch-buffers=1

[surface0]
# projection-type=4 is FISH_PERSPECTIVE (fisheye to rectilinear)
projection-type=4
surface-index=0
width={img_w}
height={img_h}

# Camera intrinsic parameters from calibration
focal-length={fx:.6f};{fy:.6f}

# Distortion coefficients (k1;k2;k3;k4)
distortion={dist_str}

# Optical center (principal point)
src-x0={cx:.6f}
src-y0={cy:.6f}
"""

    # Save nvdewarper config file
    nvdewarper_config_path = os.path.join(OUTPUT_DIRECTORY, 'nvdewarper_config.txt')
    with open(nvdewarper_config_path, 'w') as f:
        f.write(nvdewarper_config)

    print(f"\n=== nvdewarper config saved to: {nvdewarper_config_path} ===")
    print(f"focal-length={fx:.6f};{fy:.6f}")
    print(f"distortion={dist_str}")
    print(f"src-x0={cx:.6f}")
    print(f"src-y0={cy:.6f}")

    return ret, mtx, dist, rvecs, tvecs

def undistort_images(mtx, dist):
    """
    Undistort all calibration images using the calibration results.

    Args:
        mtx: Camera matrix
        dist: Distortion coefficients
    """
    if not SAVE_UNDISTORTED:
        return

    images = glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, '*.png')) + \
             glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, '*.jpg')) + \
             glob.glob(os.path.join(CALIBRATION_IMAGES_DIR, '*.jpeg'))

    if not images:
        print(f"No images found in {CALIBRATION_IMAGES_DIR}")
        return

    undistorted_dir = os.path.join(OUTPUT_DIRECTORY, 'undistorted')
    if not os.path.exists(undistorted_dir):
        os.makedirs(undistorted_dir)

    print(f"Undistorting {len(images)} images using {'FISHEYE' if USE_FISHEYE else 'STANDARD'} model...")

    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        h, w = img.shape[:2]

        if USE_FISHEYE:
            # Fisheye undistortion using remap for better quality
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                mtx, dist, np.eye(3), mtx, (w, h), cv2.CV_16SC2
            )
            dst = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        else:
            # Standard undistortion
            # Refine camera matrix based on free scaling parameter
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            # Undistort image
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

            # Crop the image (optional)
            x, y, w_roi, h_roi = roi
            if w_roi > 0 and h_roi > 0:
                dst = dst[y:y+h_roi, x:x+w_roi]

        # Save undistorted image
        output_img_path = os.path.join(undistorted_dir, f'undistorted_{os.path.basename(fname)}')
        cv2.imwrite(output_img_path, dst)

        print(f"Undistorted image {idx+1}/{len(images)}: {fname}")

    print(f"Undistorted images saved to {undistorted_dir}")

def calculate_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs):
    """
    Calculate the reprojection error for each calibration image.
    
    Args:
        objpoints: 3D points in real world space
        imgpoints: 2D points in image plane
        mtx: Camera matrix
        dist: Distortion coefficients
        rvecs: Rotation vectors
        tvecs: Translation vectors
    
    Returns:
        mean_error: Mean reprojection error
    """
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        print(f"Reprojection error for image {i+1}: {error}")
    
    mean_error = total_error / len(objpoints)
    print(f"Mean reprojection error: {mean_error}")
    
    return mean_error

def main():
    """
    Main function to run the camera calibration process.
    """
    print("Starting camera calibration...")
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = calibrate_camera()
    
    if mtx is None:
        print("Calibration failed. Exiting.")
        return
    
    # Undistort images
    undistort_images(mtx, dist)
    
    print("Camera calibration completed successfully!")

if __name__ == "__main__":
    main()