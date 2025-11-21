import cv2
import numpy as np

# ---------- 1. Read Image ----------
img = cv2.imread('sample.jpg')  # replace with your image path

if img is None:
    print("Error: Image not found.")
    exit()

print("Image shape:", img.shape)  # (height, width, channels)

# ---------- 2. Display Original Image ----------
cv2.imshow("Original", img)

# ---------- 3. Split into RGB Channels ----------
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
b, g, r = cv2.split(img)
cv2.imshow("Blue Channel", b)
cv2.imshow("Green Channel", g)
cv2.imshow("Red Channel", r)

# ---------- 4. Merge Back in Different Order ----------
merged = cv2.merge([r, g, b])  # swapped order just for demo
cv2.imshow("Merged (RGB)", merged)

# ---------- 5. Convert to Grayscale ----------
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale", gray)

# ---------- 6. Convert to HSV ----------
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("HSV", hsv)

# ---------- 7. Image Slicing (Region of Interest - ROI) ----------
# Crop a portion (e.g., top-left 1000x1000 area)
roi = img[0:1000, 0:1000]
cv2.imshow("Cropped ROI", roi)

# Modify region — e.g., make it red
img[0:1000, 0:1000] = [0, 0, 255]
cv2.imshow("Modified Image", img)

# ---------- 8. Save Result ----------
cv2.imwrite("output_gray.jpg", gray)
cv2.imwrite("output_hsv.jpg", hsv)
cv2.imwrite("output_modified.jpg", img)

# ---------- 9. Wait and Close ----------
cv2.waitKey(0)
cv2.destroyAllWindows()
