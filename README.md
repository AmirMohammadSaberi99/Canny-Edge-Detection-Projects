# Canny Edge Detection Projects

A collection of Python scripts demonstrating various applications of the Canny edge detector (and comparisons with other gradient methods) using OpenCV and Matplotlib.

## Table of Contents

1. [Canny with Fixed Thresholds](#canny-with-fixed-thresholds)
2. [Dynamic Canny via GUI Trackbars](#dynamic-canny-via-gui-trackbars)
3. [Gaussian Blur + Canny on Noisy Images](#gaussian-blur--canny-on-noisy-images)
4. [Contour Extraction on Canny Edges](#contour-extraction-on-canny-edges)
5. [Histogram Equalization vs. Canny](#histogram-equalization-vs-canny)
6. [Real‑time Canny with FPS Display](#real‑time-canny-with-fps-display)
7. [Sobel vs. Canny Comparison](#sobel-vs-canny-comparison)
8. [Masked‑Region Canny](#masked‑region-canny)

---

## Prerequisites

- Python 3.7+
- OpenCV: `pip install opencv-python`
- NumPy: `pip install numpy`
- Matplotlib: `pip install matplotlib`

---

## 1. Canny with Fixed Thresholds
**Script:** `canny_demo.py`  
**Description:** Apply Canny edge detection with thresholds 50 and 150 to two sample images and display original vs. edges.  
**Usage:**
```bash
python canny_demo.py Test.jpg Test2.jpg --th1 50 --th2 150
```

## 2. Dynamic Canny via GUI Trackbars
**Script:** `dynamic_canny.py`  
**Description:** Open a window with two trackbars to adjust Canny thresholds in real time on an image.  
**Usage:**
```bash
python dynamic_canny.py test.png
```  
Adjust sliders for “Min” and “Max”, press Esc to exit.

## 3. Gaussian Blur + Canny on Noisy Images
**Script:** `canny_blur_demo.py`  
**Description:** Reduce noise with Gaussian blur, then run Canny edge detection; shows original, blurred, and edge maps.  
**Usage:**
```bash
python canny_blur_demo.py noisy1.jpg noisy2.jpg --kernel 5 5 --sigma 1.0 --th1 50 --th2 150
```

## 4. Contour Extraction on Canny Edges
**Script:** `contour_detection.py`  
**Description:** After Canny, extract contours and overlay them on the original image.  
**Usage:**
```bash
python contour_detection.py Test.jpg Test2.jpg --th1 50 --th2 150 --kernel 5 5 --sigma 1.0
```

## 5. Histogram Equalization vs. Canny
**Script:** `hist_eq_canny_demo.py`  
**Description:** Compare edge detection before and after histogram equalization; displays original, equalized, and their Canny maps.  
**Usage:**
```bash
python hist_eq_canny_demo.py Test.jpg Test2.jpg --th1 50 --th2 150
```

## 6. Real‑time Canny with FPS Display
**Script:** `realtime_canny_fps.py`  
**Description:** Capture video (webcam or file), apply Canny in real time, and overlay current FPS on each frame.  
**Usage:**
```bash
python realtime_canny_fps.py --source 0 --th1 50 --th2 150
```

## 7. Sobel vs. Canny Comparison
**Script:** `sobel_vs_canny_demo.py`  
**Description:** Generate side‑by‑side comparison plots of Sobel gradient magnitude vs. Canny edge maps for input images.  
**Usage:**
```bash
python sobel_vs_canny_demo.py Test.jpg Test2.jpg --ksize 3 --t1 50 --t2 150
```

## 8. Masked‑Region Canny
**Script:** `canny_masked_region.py`  
**Description:** Apply Canny edge detection only within a circular mask region, overlay edges on the original.  
**Usage:**
```bash
python canny_masked_region.py Test.jpg Test2.jpg --t1 50 --t2 150
```

---

## License

This collection is released under the MIT License. Feel free to use and adapt these scripts for your own edge‑detection experiments.

