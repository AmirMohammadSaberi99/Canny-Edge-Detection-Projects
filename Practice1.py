import cv2
import matplotlib.pyplot as plt

# Paths to the two images
image_paths = ['Test.jpg', 'Test2.jpg']
titles = ['Test.jpg', 'Test2.jpg']

# Parameters for Canny
threshold1, threshold2 = 50, 150

# Set up plotting: 2 rows, 2 columns
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for i, path in enumerate(image_paths):
    # Load the image in grayscale
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not load image at {path}")
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, threshold1, threshold2)
    
    # Show original
    ax_orig = axes[i, 0]
    ax_orig.imshow(gray, cmap='gray')
    ax_orig.set_title(f"{titles[i]} - Original")
    ax_orig.axis('off')
    
    # Show edges
    ax_edge = axes[i, 1]
    ax_edge.imshow(edges, cmap='gray')
    ax_edge.set_title(f"{titles[i]} - Canny (50,150)")
    ax_edge.axis('off')

plt.tight_layout()
plt.show()
