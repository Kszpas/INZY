# 6️⃣ TEST NA POJEDYNCZYM PRZYKŁADZIE (opcjonalnie)
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import cv2 as cv
import numpy as np

model = load_model('models/field_classifier_final_balanced.keras')
classes = ['empty', 'white', 'black']

img = cv.imread('1.jpg')
img = cv.resize(img, (64, 64))
img = img.astype('float32') / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)[0]
print("✅ Wynik:", classes[np.argmax(pred)], f"(pewność {max(pred)*100:.1f}%)")

