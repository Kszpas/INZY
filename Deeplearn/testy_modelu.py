import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# 1️⃣ Utwórz generator z tymi samymi ustawieniami co przy treningu
datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# to ważne – wczytujemy z folderu 'train' lub 'val', żeby odtworzyć kolejność klas:
train_gen = datagen.flow_from_directory(
    'dataset/train',
    target_size=(64,64),
    batch_size=1,
    class_mode='categorical'
)

# 2️⃣ Wczytaj gotowy model
model = tf.keras.models.load_model('models/field_classifier_final_balanced.keras')

# 3️⃣ Wczytaj pojedynczy obraz testowy
img_path = 'dataset/val/white/white_00001.jpg'
img = image.load_img(img_path, target_size=(64,64))
img_array = np.expand_dims(image.img_to_array(img) / 255.0, axis=0)

# 4️⃣ Predykcja
pred = model.predict(img_array)
pred_idx = np.argmax(pred)
# Znajdź nazwę klasy po indeksie:
pred_label = list(train_gen.class_indices.keys())[list(train_gen.class_indices.values()).index(pred_idx)]

# 5️⃣ Pokaż wynik
plt.imshow(img)
plt.title(f"Predicted: {pred_label}")
plt.axis('off')
plt.show()

print(f"Predykcja: {pred_label}")

print(train_gen.class_indices)