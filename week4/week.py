import cv2
from datetime import datetime

modelfile = "res10_300x300_ssd_iter_140000.caffemodel"
configfile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configfile, modelfile)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (1280, 720))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        break

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]

    # Face Detection
    resized_image = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * [w, h, w, h]
            (x, y, x1, y1) = box.astype("int")
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)

    # Sobel + Grayscale + Overlays
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sobel_y = cv2.convertScaleAbs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))
    combined = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(frame, f'Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f'Resolution: {w}x{h}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Live Feed", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
