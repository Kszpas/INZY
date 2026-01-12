import tensorflow as tf
import cv2
import numpy as np

model = tf.keras.models.load_model("models/field_classifier_final_balanced.keras")
tile = cv2.imread("empty_00195.jpg")
tile = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
tile = cv2.resize(tile, (128,128)).astype(np.float32)/255.0
pred = model.predict(np.expand_dims(tile, 0))
print("Pred:", pred)
print("Klasa:", np.argmax(pred))