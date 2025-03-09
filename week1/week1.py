# Computer Vision Week 1
#pip install opencv-python
# Computer vision is a field of artificial intelligence that focuses on enabling computers to interpret and understand visual information from the world. It involves developing algorithms and models that can extract useful information from images, videos, and other visual data. This information can then be used to make predictions, classify objects, detect anomalies, and more. Some examples of computer vision applications include image recognition, object detection, image segmentation, and image processing.
#OpenCV is a computer vision libary that connects the computer with vision devices

import cv2
print("OpenCV version:", cv2.__version__)


# Read and display an image
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
img = cv2.imread("week1/leaf.jpg")
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
