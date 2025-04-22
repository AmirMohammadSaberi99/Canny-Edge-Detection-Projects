#!/usr/bin/env python3
"""
sobel_vs_canny_demo.py

Compares Sobel gradient magnitude vs. Canny edge detection on one or more
grayscale images. Displays each image’s original, Sobel magnitude, and Canny
edge map side by side.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def compute_sobel_magnitude(gray: np.ndarray, ksize: int) -> np.ndarray:
    """
    Compute the gradient magnitude of a grayscale image using the Sobel operator.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel grayscale image.
    ksize : int
        Sobel kernel size (must be odd).

    Returns
    -------
    np.ndarray
        8‑bit gradient magnitude image.
    """
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    mag = np.sqrt(dx*dx + dy*dy)
    mag = np.clip((mag / mag.max()) * 255, 0, 255).astype(np.uint8)
    return mag

def compute_canny_edges(gray: np.ndarray, t1: int, t2: int) -> np.ndarray:
    """
    Compute the Canny edge map of a grayscale image.

    Parameters
    ----------
    gray : np.ndarray
        Single-channel grayscale image.
    t1 : int
        Lower threshold for the hysteresis procedure.
    t2 : int
        Upper threshold for the hysteresis procedure.

    Returns
    -------
    np.ndarray
        8‑bit binary edge map.
    """
    return cv2.Canny(gray, t1, t2)

def plot_comparison(image_paths, ksize, t1, t2):
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, path in enumerate(image_paths):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Error: could not load '{path}'", file=sys.stderr)
            continue

        sobel_mag = compute_sobel_magnitude(gray, ksize)
        canny_edges = compute_canny_edges(gray, t1, t2)

        # Original
        ax = axes[i, 0]
        ax.imshow(gray, cmap='gray')
        ax.set_title(f"{path}\nOriginal")
        ax.axis('off')

        # Sobel magnitude
        ax = axes[i, 1]
        ax.imshow(sobel_mag, cmap='gray')
        ax.set_title(f"Sobel magnitude\nksize={ksize}")
        ax.axis('off')

        # Canny edges
        ax = axes[i, 2]
        ax.imshow(canny_edges, cmap='gray')
        ax.set_title(f"Canny edges\n({t1},{t2})")
        ax.axis('off')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(
        description="Compare Sobel magnitude vs. Canny edges on images."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to one or more grayscale images (e.g. Test.jpg Test2.jpg)"
    )
    parser.add_argument(
        "--ksize", "-k",
        type=int,
        default=3,
        help="Sobel kernel size (odd integer, default=3)"
    )
    parser.add_argument(
        "--t1",
        type=int,
        default=50,
        help="Lower threshold for Canny (default=50)"
    )
    parser.add_argument(
        "--t2",
        type=int,
        default=150,
        help="Upper threshold for Canny (default=150)"
    )

    args = parser.parse_args()
    plot_comparison(args.images, args.ksize, args.t1, args.t2)

if __name__ == "__main__":
    main()
