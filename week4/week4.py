import cv2
import numpy

# Load the pretrained model
# Pretrained are highly generalised and may not work for specific tasks
# false positive = wrong prediction and false positive has more likeness on pretrained model

#harcascade is one of world's first pretrained model for facial recognition that uses 5 filters

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
image =cv2.imread('2face.jpg')

# harfascade uses black and white so we need tp convert the photo to gray

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('Gray Image', cv2.WINDOW_NORMAL)
cv2.imshow('Gray Image', gray)

# Detect faces in the image
detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                               flags=cv2.CASCADE_SCALE_IMAGE)
print((detected_faces))

for (x,y,w,h) in detected_faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2) #color of bounding box


cv2.namedWindow('Haar Fascade Face Detection', cv2.WINDOW_NORMAL)
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
