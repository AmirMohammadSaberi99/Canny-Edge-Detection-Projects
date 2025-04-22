#!/usr/bin/env python3
"""
dynamic_canny.py

Loads a grayscale image, then provides a GUI with a single
1920×1080 window and two trackbars to adjust the Canny thresholds
in real time. Press Esc to exit.
"""

import cv2
import sys
import argparse

def nothing(x):
    pass

def main(image_path: str, window_w: int = 1920, window_h: int = 1080):
    # 1. Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: could not load '{image_path}'", file=sys.stderr)
        sys.exit(1)

    # 2. Make a resizable window at the desired resolution
    win = 'Canny Edge (Min/Max)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, window_w, window_h)

    # 3. Create two trackbars in that window
    cv2.createTrackbar('Min', win, 50, 500, nothing)
    cv2.createTrackbar('Max', win, 150, 500, nothing)

    while True:
        # 4. Read thresholds from trackbars
        t_min = cv2.getTrackbarPos('Min', win)
        t_max = cv2.getTrackbarPos('Max', win)

        # 5. Apply Canny
        edges = cv2.Canny(img, t_min, t_max)

        # 6. Stack original and edges side by side
        combined = cv2.hconcat([
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
            cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        ])

        # 7. Show in the same window (which is already sized to 1920×1080)
        cv2.imshow(win, combined)

        # 8. Exit on Esc
        if cv2.waitKey(30) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dynamic Canny with fixed 1920x1080 window and trackbars."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="test.jpg",
        help="Path to a grayscale image (default: test.png)"
    )
    args = parser.parse_args()
    main(args.image_path)
