#!/usr/bin/env python3
"""
canny_masked_region.py

Loads one or more grayscale images, applies Canny edge detection only within
a circular mask region, and visualizes the original, masked region, and
edges-overlaid result for each image.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def apply_canny_masked(
    gray: np.ndarray,
    t1: int,
    t2: int,
    mask_center: tuple[int, int],
    mask_radius: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply a circular mask and then Canny edge detection.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale input image.
    t1, t2 : int
        Lower and upper thresholds for Canny.
    mask_center : (x, y)
        Center of the circular mask.
    mask_radius : int
        Radius of the circular mask.

    Returns
    -------
    masked_gray : np.ndarray
        Grayscale image with everything outside the mask zeroed.
    overlay : np.ndarray
        BGR image with edges (in red) overlaid on the original.
    """
    h, w = gray.shape
    mask = np.zeros_like(gray)
    cv2.circle(mask, mask_center, mask_radius, 255, -1)

    # Mask the grayscale image
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Canny on masked region
    edges = cv2.Canny(masked_gray, t1, t2)

    # Create overlay: convert gray to BGR, draw red edges
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[edges != 0] = (0, 0, 255)  # Red in BGR

    return masked_gray, overlay

def main(
    image_paths: list[str],
    t1: int,
    t2: int
):
    fig, axes = plt.subplots(len(image_paths), 3, figsize=(15, 5*len(image_paths)))
    if len(image_paths) == 1:
        axes = np.expand_dims(axes, 0)

    for idx, path in enumerate(image_paths):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Error: could not load '{path}'", file=sys.stderr)
            continue

        h, w = gray.shape
        center = (w // 2, h // 2)
        radius = min(h, w) // 4

        masked_gray, overlay = apply_canny_masked(gray, t1, t2, center, radius)

        # Original
        ax = axes[idx, 0]
        ax.imshow(gray, cmap='gray')
        ax.set_title(f"{path}\nOriginal")
        ax.axis('off')

        # Masked region
        ax = axes[idx, 1]
        ax.imshow(masked_gray, cmap='gray')
        ax.set_title("Masked Region")
        ax.axis('off')

        # Overlay edges
        ax = axes[idx, 2]
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax.set_title("Edges on Mask")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply Canny edge detection only on a masked region."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to grayscale images (e.g. Test.jpg Test2.jpg)"
    )
    parser.add_argument(
        "--t1",
        type=int,
        default=50,
        help="Lower Canny threshold (default=50)"
    )
    parser.add_argument(
        "--t2",
        type=int,
        default=150,
        help="Upper Canny threshold (default=150)"
    )
    args = parser.parse_args()

    main(args.images, args.t1, args.t2)
