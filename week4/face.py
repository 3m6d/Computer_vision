  # https://shorturl.at/gipNr
"""
  1. We are using pytorch-cpu version only for this workshop in order to download the torch's resources quickly.
  To download the cpu only version the Pytorch library, go to: https://pytorch.org/ and in the Compute Platform
  option select 'cpu'. Then copy the command given in the 'Run this Command' section and install it inside your
  environment.
  If you want to leverage the GPU: simply to pip install torch in the terminal after activating the environment.
  """

import cv2
import torch
import sqlite3
import numpy as np
import os
import torchvision.models as models
import torchvision.transforms as T
from numpy.linalg import norm

"""
Using ResNet architecture to extract 2048 dimension's image embedding.
"""
resnet = models.resnet50(pretrained=True)
resnet.fc = torch.nn.Identity()  # Use as feature extractor
resnet.eval()

"""
This is image preprocessing that we studied in our week 2. 
"""
transform = T.Compose([
      T.ToPILImage(),
      T.Resize((224,224)),
      T.ToTensor(),
  ])

face_net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

conn = sqlite3.connect('embeddings.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS faces
               (id INTEGER PRIMARY KEY, name TEXT, embedding BLOB)''')


def get_embedding(img):
    with torch.no_grad():
        img_tensor = transform(img).unsqueeze(0)
        embedding = resnet(img_tensor).numpy().flatten()
        embedding = embedding / norm(embedding)  # Normalize embedding
    return embedding


def save_known_faces(directory='faces'):
    c.execute('DELETE FROM faces')
    for idx, filename in enumerate(os.listdir(directory)):
        path = os.path.join(directory, filename)
        img = cv2.imread(path)
        face = detect_single_face(img)
        if face is not None:
            embedding = get_embedding(face)
            name = os.path.splitext(filename)[0]
            c.execute("INSERT INTO faces (id, name, embedding) VALUES (?, ?, ?)",
                      (idx, name, embedding.tobytes()))
            print(f"Stored embedding for {name}")
    conn.commit()


def load_known_faces():
    c.execute("SELECT id, name, embedding FROM faces")
    return [(id, name, np.frombuffer(embedding, dtype=np.float32))
            for id, name, embedding in c.fetchall()]


def calculate_distance(emb1, emb2):
    return norm(emb1 - emb2)


def distance_to_percentage(dist, threshold=1.0):
    percentage = max(0, 100 * (1 - dist / threshold))
    return round(percentage, 2)


def detect_single_face(img, confidence_threshold=0.7):
    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    if detections.shape[2] > 0:
        confidence = detections[0, 0, 0, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            face_img = img[y1:y2, x1:x2]
            return face_img
    return None


def recognize_face_webcam():
    known_faces = load_known_faces()
    cap = cv2.VideoCapture(0)
    detection_confidence = 0.7
    recognition_threshold = 0.8  # Adjust based on performance

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        face_net.setInput(blob)
        detections = face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > detection_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)

                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue

                embedding = get_embedding(face_img)

                distances = [(id, name, calculate_distance(embedding, emb))
                             for id, name, emb in known_faces]
                best_match = min(distances, key=lambda x: x[2])
                matched_id, matched_name, distance = best_match

                closeness = distance_to_percentage(distance, recognition_threshold)
                label = f"{matched_name} ({closeness}%) Dist:{distance:.2f}"

                color = (0, 255, 0) if distance < recognition_threshold else (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.namedWindow("Recognized face", cv2.WINDOW_NORMAL)
        cv2.imshow("Recognized face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    save_known_faces()      # Run once to store embeddings
    recognize_face_webcam() # Start real-time recognitiom

