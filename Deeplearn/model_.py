
#   TRAINING FIELD CLASSIFIER (CNN)



import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1️⃣ DEFINICJA MODELU CNN

def create_cnn_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
        MaxPooling2D((2,2)),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 klasy: empty / white / black
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model



# 2️⃣ PRZYGOTOWANIE DANYCH

train_path = 'dataset/train'
val_path = 'dataset/val'

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    val_path,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)


# 3️⃣ TRENING MODELU

model = create_cnn_model()

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20  # możesz dostosować po wynikach
)


# 4️⃣ ZAPIS MODELU

os.makedirs('models', exist_ok=True)
model.save('models/field_classifier.h5')
print("✅ Model zapisany w 'models/field_classifier.h5'")


# 5️⃣ WYKRES DOKŁADNOŚCI

plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Trening')
plt.plot(history.history['val_accuracy'], label='Walidacja')
plt.title("Dokładność modelu CNN")
plt.xlabel("Epoki")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


