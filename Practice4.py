#!/usr/bin/env python3
"""
contour_detection.py

Loads one or more images, denoises with Gaussian blur, applies Canny edge detection,
extracts contours, and overlays them on the original images. Displays original vs.
contours for each input.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def process_image(
    image_path: str,
    gaussian_kernel: tuple[int, int],
    sigma: float,
    canny_thresh1: int,
    canny_thresh2: int,
    contour_mode: int,
    contour_method: int,
    draw_color: tuple[int, int, int],
    thickness: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load image, blur, detect edges, find contours, draw them.

    Returns (original_rgb, contoured_rgb) as two RGB arrays.
    """
    # 1. Load in BGR
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not load image '{image_path}'")
    # 2. Convert to grayscale and blur
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, gaussian_kernel, sigma)
    # 3. Canny edge detection
    edges = cv2.Canny(blurred, canny_thresh1, canny_thresh2)
    # 4. Find contours
    contours, _ = cv2.findContours(edges, contour_mode, contour_method)
    # 5. Draw contours on a copy of original
    overlay = img_bgr.copy()
    cv2.drawContours(overlay, contours, -1, draw_color, thickness)
    # 6. Convert both to RGB for plotting
    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    contoured_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return orig_rgb, contoured_rgb

def main(args):
    # Prepare subplots: one row per image, 2 cols (orig, contoured)
    n = len(args.images)
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    # Process each
    for i, img_path in enumerate(args.images):
        try:
            orig, contoured = process_image(
                img_path,
                tuple(args.kernel),
                args.sigma,
                args.thresh1,
                args.thresh2,
                getattr(cv2, args.contour_mode),
                getattr(cv2, args.contour_method),
                tuple(args.color),
                args.thickness
            )
        except FileNotFoundError as e:
            print(e, file=sys.stderr)
            continue

        # Plot original
        ax0 = axes[i, 0]
        ax0.imshow(orig)
        ax0.set_title(f"{img_path}\nOriginal")
        ax0.axis('off')

        # Plot with contours
        ax1 = axes[i, 1]
        ax1.imshow(contoured)
        ax1.set_title(f"{img_path}\nContours over Canny")
        ax1.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Denoise, Canny-edge, and overlay contours on images."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to input images (e.g. Test.jpg Test2.jpg)"
    )
    parser.add_argument(
        "--kernel", "-k",
        type=int,
        nargs=2,
        default=(5, 5),
        metavar=("KX", "KY"),
        help="Gaussian kernel size (odd ints), default=5 5"
    )
    parser.add_argument(
        "--sigma", "-s",
        type=float,
        default=1.0,
        help="Gaussian sigma (default=1.0)"
    )
    parser.add_argument(
        "--thresh1", "-t1",
        type=int,
        default=50,
        help="Canny lower threshold (default=50)"
    )
    parser.add_argument(
        "--thresh2", "-t2",
        type=int,
        default=150,
        help="Canny upper threshold (default=150)"
    )
    parser.add_argument(
        "--contour_mode",
        type=str,
        default="RETR_EXTERNAL",
        choices=["RETR_EXTERNAL", "RETR_LIST", "RETR_CCOMP", "RETR_TREE"],
        help="Contour retrieval mode (default: RETR_EXTERNAL)"
    )
    parser.add_argument(
        "--contour_method",
        type=str,
        default="CHAIN_APPROX_SIMPLE",
        choices=["CHAIN_APPROX_NONE", "CHAIN_APPROX_SIMPLE", "CHAIN_APPROX_TC89_L1", "CHAIN_APPROX_TC89_KCOS"],
        help="Contour approximation method (default: CHAIN_APPROX_SIMPLE)"
    )
    parser.add_argument(
        "--color", "-c",
        type=int,
        nargs=3,
        default=(255, 0, 0),
        metavar=("B", "G", "R"),
        help="Contour color in BGR (default: 255 0 0)"
    )
    parser.add_argument(
        "--thickness", "-t",
        type=int,
        default=2,
        help="Contour line thickness (default: 2)"
    )

    args = parser.parse_args()
    main(args)
