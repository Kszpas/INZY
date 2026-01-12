import sklearn
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import numpy as np
import matplotlib.pyplot as plt
import os

#  PARAMETRY
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 25

#GENERATORY DANYCH z AUGMENTACJĄ
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.25,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Kolejność klas:", train_gen.class_indices)

#  OBLICZANIE CLASS WEIGHT AUTOMATYCZNIE
auto_class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
auto_class_weights = dict(enumerate(auto_class_weights))
print("Wagi automatyczne (compute_class_weight):", auto_class_weights)

# 
#  TUTAJ MOŻESZ RĘCZNIE ZMIENIAĆ WAGI KLAS:
#    Kolejność odpowiada: {'black': 0, 'empty': 1, 'white': 2}
#    Możesz dostosować wartości, jeśli np. model nadal źle rozpoznaje "white".
#    Przykład:
class_weights = {
    0: 2.2,   # black
    1: 1.0,   # empty
    2: 3.0    # white
}
#  Zmieniaj te liczby, jeśli chcesz przetestować inne proporcje
#    - zwiększ wartość, jeśli klasa jest trudna / rzadka
#    - zmniejsz, jeśli jest bardzo częsta
# 

#  DEFINICJA MODELU
model = tf.keras.models.load_model('models/field_classifier_final_balanced.keras')

#  KOMPILACJA MODELU
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#  CALLBACKS
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True),
    ModelCheckpoint('models/field_classifier_balanced.keras', save_best_only=True)
]

#  TRENING
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=class_weights,  #  używamy ręcznie ustawionych wag
    callbacks=callbacks
)

#  ZAPIS MODELU
model.save('models/field_classifier_final_balanced.keras')
print("\n✅ Model wytrenowany i zapisany jako field_classifier_final_balanced.keras")

#  WYKRESY TRENINGU
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Dokładność (accuracy)')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Strata (loss)')
plt.legend()
plt.show()