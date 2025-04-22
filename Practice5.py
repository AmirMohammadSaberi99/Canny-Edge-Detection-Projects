#!/usr/bin/env python3
"""
hist_eq_canny_demo.py

Compare Canny edge detection results before and after histogram equalization
on one or more grayscale images.
"""

import cv2
import matplotlib.pyplot as plt
import argparse
import sys

def canny_before_after(image_gray, thresh1, thresh2):
    """
    Perform Canny edge detection before and after histogram equalization.

    Parameters
    ----------
    image_gray : np.ndarray
        Input grayscale image.
    thresh1, thresh2 : int
        Lower and upper thresholds for Canny.

    Returns
    -------
    tuple (orig, equalized, edges_before, edges_after)
    """
    # Original
    orig = image_gray
    # Equalize histogram
    eq = cv2.equalizeHist(orig)
    # Canny before and after
    edges_before = cv2.Canny(orig, thresh1, thresh2)
    edges_after  = cv2.Canny(eq,   thresh1, thresh2)
    return orig, eq, edges_before, edges_after

def main(images, thresh1, thresh2):
    n = len(images)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = [axes]

    for i, path in enumerate(images):
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"Error: could not load '{path}'", file=sys.stderr)
            continue

        orig, eq, before, after = canny_before_after(gray, thresh1, thresh2)

        cols = axes[i]
        titles = ["Original", "Equalized", f"Canny Before ({thresh1},{thresh2})", f"Canny After ({thresh1},{thresh2})"]
        images = [orig, eq, before, after]

        for ax, img, title in zip(cols, images, titles):
            ax.imshow(img, cmap='gray')
            ax.set_title(f"{path}\n{title}")
            ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Canny edges before/after histogram equalization."
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Paths to one or more grayscale images (e.g. Test.jpg Test2.jpg)"
    )
    parser.add_argument(
        "--thresh1",
        type=int,
        default=50,
        help="Lower threshold for Canny (default=50)"
    )
    parser.add_argument(
        "--thresh2",
        type=int,
        default=150,
        help="Upper threshold for Canny (default=150)"
    )

    args = parser.parse_args()
    main(args.images, args.thresh1, args.thresh2)
