import cv2
import matplotlib.pyplot as plt
import os

image_path_2 = "../week2/RGBA.png"
original_image = cv2.imread(image_path_2, cv2.IMREAD_UNCHANGED)


original_image_shape = original_image.shape  # Get the shape of the image
print("Original image shape: ", original_image_shape)

# Show the image using OpenCV
cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.imshow("Original", original_image)

# OpenCV reads RGBA as BGRA, so we need to convert it to RGB
plt.imshow(original_image)
plt.title("Original image (Possibly BGRA)")

# Ensure the saving directory exists
save_dir = "../week2/"
os.makedirs(save_dir, exist_ok=True)

plt.savefig(os.path.join(save_dir, "BGRA.png"))
plt.show()

# Convert BGRA to RGB
if original_image.shape[-1] == 4:  # If the image has 4 channels (BGRA)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGRA2RGB)
else:
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

cv2.imwrite(os.path.join(save_dir, "RGB.png"), original_image)

# Convert to Grayscale
gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
print("Grayscale image shape: ", gray_image.shape)
cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
cv2.imshow("Grayscale", gray_image)

cv2.imwrite(os.path.join(save_dir, "Grayscale.png"), gray_image)

print("Grayscale image shape: ", gray_image.shape)
print("RGB image shape: ", original_image.shape)


#To convert grayscale to Black and White
#binarization
_, thresholded = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
cv2.imshow("Thresholded", thresholded)

cv2.imwrite(os.path.join(save_dir, "BnW.png"), thresholded)


#binarization with inverse
_, thresholded_inverse= cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY_INV)
cv2.namedWindow("Thresholded", cv2.WINDOW_NORMAL)
cv2.imshow("Thresholded", thresholded_inverse)

cv2.imwrite(os.path.join(save_dir, "BnWInverse.png"), thresholded_inverse)

## Image Resizing is very important in image prccessing
resized_image = cv2.resize(original_image, (300, 300))
cv2.namedWindow("Resized", cv2.WINDOW_NORMAL)
cv2.imshow("Resized", resized_image)

# Save the resized image
cv2.imwrite(os.path.join(save_dir, "Resized.png"), resized_image)

# ROI = Region of Interest which means image slicing or image cropping

# Define the region of interest (ROI)]


#flip horizontally and vertically 
flip_horizontal = cv2.flip(original_image, 1) #1 is horizontal
flip_vertical = cv2.flip(original_image, 0) #0 is vertical

cv2.namedWindow("Flip Horizontal", cv2.WINDOW_NORMAL)
cv2.imshow("Flip Horizontal", flip_horizontal)
cv2.imwrite(os.path.join(save_dir, "FlipHorizontal.png"), flip_horizontal)

cv2.namedWindow("Flip Vertical", cv2.WINDOW_NORMAL)
cv2.imshow("Flip Vertical", flip_vertical)
cv2.imwrite(os.path.join(save_dir, "FlipVertical.png"), flip_vertical)


#rotate 90 degrees clockwise
rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)

cv2.namedWindow("Rotated", cv2.WINDOW_NORMAL)
cv2.imshow("Rotated", rotated_image)
cv2.imwrite(os.path.join(save_dir, "Rotated.png"), rotated_image)

#brightness
Converted_image = cv2.convertScaleAbs(original_image, alpha=1.5, beta=2) #alpha is brightness and beta is contrast
cv2.namedWindow("Brightness", cv2.WINDOW_NORMAL)
cv2.imshow("Brightness", Converted_image)
cv2.imwrite(os.path.join(save_dir, "Brightness.png"), Converted_image)


#dedicated contrast
gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY) # cvt is convert
equ = cv2.equalizeHist(gray) #equalize histogram with contrast to reduce noise
cv2.namedWindow("Contrast", cv2.WINDOW_NORMAL)
cv2.imshow("Contrast", equ)
cv2.imwrite(os.path.join(save_dir, "Contrast.png"), equ)

# Calculate the mean and standard deviation of the grayscale image  


# Wait for key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
