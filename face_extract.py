import cv2
from mtcnn import MTCNN
import os

detector = MTCNN()

def extract_face(image_path):
    img = cv2.imread(image_path)
    results = detector.detect_faces(img)

    if results:
        x, y, w, h = results[0]['box']
        face = img[y:y+h, x:x+w]
        return face
    return None