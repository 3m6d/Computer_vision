import cv2
import numpy as np

img = "../week1/leaf.jpg"

image = cv2.imread(img)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Gray Image", gray_image)
cv2.waitKey(0)

image = cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.imshow("Image", image)
cv2.waitKey(0)
