import cv2
import numpy

modelfile = "res10_300x300_ssd_iter_140000.caffemodel"
configfile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configfile, modelfile)

image = cv2.imread("2face.jpg") 

(h,m) = image.shape[:2] #slicing height width and channel

resized_image = cv2.resize(image,(300,300)) #very important

blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))

net.setInput(blob)

detections = net.forward()
print(detections)

for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5: 
        box = detections[0, 0, i, 3:7] * [m, h, m, h]
        (x,y,x1,y1) = box.astype("int") 
        cv2.rectangle(image,(x,y),(x1,y1),(0,0,255),2)

cv2.namedWindow('Face detection', cv2.WINDOW_NORMAL)
cv2.imshow('Face detection, image', image)
cv2.waitKey(0)  
cv2.destroyAllWindows()