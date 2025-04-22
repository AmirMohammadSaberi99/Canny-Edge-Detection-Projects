#!/usr/bin/env python3
"""
canny_blur_demo.py

Loads one or more grayscale images, applies Gaussian blur to reduce noise,
then runs Canny edge detection and displays the original, blurred, and
edge‑detected images side by side for each input.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

def detect_edges_with_blur(
    image_gray: np.ndarray,
    kernel_size: tuple[int, int],
    sigma: float,
    thresh1: int,
    thresh2: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply Gaussian blur then Canny edge detection.

    Parameters
    ----------
    image_gray : np.ndarray
        Single‑channel input image.
    kernel_size : (int, int)
        Size of the Gaussian kernel (odd, e.g. (5,5)).
    sigma : float
        Standard deviation for Gaussian blur.
    thresh1 : int
        Lower threshold for Canny.
    thresh2 : int
        Upper threshold for Canny.

    Returns
    -------
    blurred : np.ndarray
        The blurred image.
    edges : np.ndarray
        Binary edge map from Canny.
    """
    # 1. Gaussian blur
    blurred = cv2.GaussianBlur(image_gray, kernel_size, sigma)
    # 2. Canny edge detection
    edges = cv2.Canny(blurred, thresh1, thresh2)
    return blurred, edges

def main(
    image_paths: list[str],
    kernel_size: tuple[int, int],
    sigma: float,
    thresh1: int,
    thresh2: int
):
    n = len(image_paths)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i, path in enumerate(image_paths):
        # Load grayscale
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Error: could not load '{path}'", file=sys.stderr)
            continue

        # Detect edges after blur
        blurred, edges = detect_edges_with_blur(
            gray, kernel_size, sigma, thresh1, thresh2
        )

        # Titles and images for plotting
        imgs = [gray, blurred, edges]
        titles = ["Original", f"Blurred {kernel_size}, σ={sigma}", f"Canny {thresh1},{thresh2}"]

        for j in range(3):
            ax = axes[i, j]
            ax.imshow(imgs[j], cmap='gray')
            ax.set_title(f"{path}\n{titles[j]}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply Gaussian blur + Canny edge detection on noisy images."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to grayscale images (e.g. noisy1.jpg noisy2.jpg)"
    )
    parser.add_argument(
        "--kernel", "-k",
        type=int,
        nargs=2,
        default=(5, 5),
        metavar=("KX","KY"),
        help="Gaussian kernel size (odd ints), default=5 5"
    )
    parser.add_argument(
        "--sigma", "-s",
        type=float,
        default=1.0,
        help="Gaussian sigma (standard deviation), default=1.0"
    )
    parser.add_argument(
        "--thresh1", "-t1",
        type=int,
        default=50,
        help="Lower Canny threshold, default=50"
    )
    parser.add_argument(
        "--thresh2", "-t2",
        type=int,
        default=150,
        help="Upper Canny threshold, default=150"
    )
    args = parser.parse_args()

    main(
        args.images,
        tuple(args.kernel),
        args.sigma,
        args.thresh1,
        args.thresh2
    )
