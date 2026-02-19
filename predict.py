import os
print(os.listdir())
import tensorflow as tf
import cv2
import numpy as np
import sys

IMG_SIZE = 128

model = tf.keras.models.load_model("deepfake_model.h5")

def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        print("❌ Image not found! Check file name.")
        return

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        print("🟥 FAKE IMAGE")
    else:
        print("🟩 REAL IMAGE")

    print("Confidence:", float(prediction))

# Take input from command line
if len(sys.argv) > 1:
    predict_image(sys.argv[1])
else:
    predict_image("test.jpg")