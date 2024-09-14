Quiz 3: Introduction to OpenCV - Sigma ML
This project demonstrates a variety of image processing techniques using OpenCV and matplotlib in Python. The focus is on loading and manipulating images, applying filters, performing thresholding, and using morphological operations to process images effectively.

Table of Contents
Introduction
Dependencies
Image Loading and Conversion
Thresholding
Binary Thresholding
Otsu Thresholding
Image Smoothing
Morphological Operations
Feature Extraction - Edge Detection
Conclusion
Introduction
This project utilizes OpenCV and matplotlib to explore various image processing operations, including:

Image loading and color conversion (BGR to RGB, GRAYSCALE)
Histogram equalization to enhance contrast
Binary and Otsu thresholding techniques
Smoothing operations like Gaussian, Median, and Mean blurring
Morphological operations like erosion, dilation, and closing
Feature extraction using edge detection (Canny edge detector)
Dependencies
Before running the notebook, ensure you have the following dependencies installed:

bash
Copy code
pip install opencv-python matplotlib numpy
Additionally, the notebook uses Google Colab patches for displaying images:

bash
Copy code
pip install google-colab
Image Loading and Conversion
The project demonstrates loading an image in grayscale and converting it into RGB:

python
Copy code
img = cv.imread('portrait_lady.png', cv.IMREAD_GRAYSCALE)
img_rgb = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
plt.imshow(img_rgb)
plt.show()
It also explains the differences between OpenCV’s BGR format and matplotlib's RGB format.

Thresholding
Binary Thresholding
Binary thresholding involves converting an image into a binary scale, differentiating between background and foreground pixels based on a threshold value.

python
Copy code
ret, bin_img_thresh = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
cv2_imshow(bin_img_thresh)
Otsu Thresholding
Otsu's method automatically calculates the optimal threshold value to separate background and foreground.

python
Copy code
ret_otsu, otsu_thresh_img = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
Image Smoothing
Image smoothing techniques, such as Gaussian, Median, and Mean blurring, are applied to reduce noise in the image.

python
Copy code
# Gaussian blurring
gaussian_blur = cv.GaussianBlur(img, (7, 7), 0)
Each smoothing method offers different noise reduction effects while preserving edge details to varying degrees.

Morphological Operations
Morphological operations such as erosion, dilation, opening, and closing are applied to manipulate the structures within an image.

python
Copy code
kernel = np.ones((3, 3), np.uint8)
erosion = cv.erode(img, kernel, iterations=1)
These operations are particularly useful in tasks like foreground-background separation, noise reduction, and joining broken parts of objects.

Feature Extraction - Edge Detection
Finally, Canny edge detection is applied to detect the edges and contours in the image:

python
Copy code
edges = cv.Canny(img, 100, 200)
plt.subplot(121), plt.imshow(edges, cmap='gray')
Edge detection is critical for feature extraction and object detection in many computer vision applications.

Conclusion
This project provides a basic introduction to image processing using OpenCV. By performing these operations, we can effectively preprocess images for further analysis, object detection, and recognition.
