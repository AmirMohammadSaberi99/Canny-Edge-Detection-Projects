#!/usr/bin/env python3
"""
realtime_canny_fps.py

Capture video from webcam (or file), apply Canny edge detection to each frame,
display the result with a live FPS counter. Press 'q' to exit.
"""

import cv2
import time
import argparse
import sys

def main(source=0, thresh1=50, thresh2=150):
    # 1. Open video source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: unable to open video source {source}", file=sys.stderr)
        sys.exit(1)

    prev = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. Compute FPS
        now = time.time()
        fps = 1.0 / (now - prev) if prev else 0.0
        prev = now

        # 3. Convert to grayscale and run Canny
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, thresh1, thresh2)

        # 4. Overlay FPS on the edges image
        disp = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            disp,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # 5. Show
        cv2.imshow("Canny Edges + FPS", disp)

        # 6. Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real‑time Canny edge detection with FPS counter."
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="0",
        help="Video source (camera index or file path). Default=0 (webcam)"
    )
    parser.add_argument(
        "--th1", "-1",
        type=int,
        default=50,
        help="Canny lower threshold (default=50)"
    )
    parser.add_argument(
        "--th2", "-2",
        type=int,
        default=150,
        help="Canny upper threshold (default=150)"
    )
    args = parser.parse_args()

    # allow numeric camera index
    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    main(src, args.th1, args.th2)
