import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import class_weight
import matplotlib.pyplot as plt

# ===========================================
# 📁 PARAMETRY
# ===========================================
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR   = os.path.join(BASE_DIR, "val")

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 8  # krótkie, bezpieczne douczanie

# ===========================================
# 🧠 WCZYTANIE ISTNIEJĄCEGO MODELU
# ===========================================
MODEL_PATH = 'models/field_classifier_final_balanced.keras'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Nie znaleziono pliku modelu: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model wczytany:", MODEL_PATH)

# ===========================================
# 🔧 USTAWIENIA AUGMENTACJI
# ===========================================

# Umiarkowana augmentacja: drobne rotacje i jasność, bez odwracania planszy
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    brightness_range=[0.9, 1.1],
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("📊 Kolejność klas:", train_gen.class_indices)

# ===========================================
# ⚖️ WYZNACZENIE CLASS_WEIGHTS (automatycznie)
# ===========================================
auto_class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_gen.classes),
    y=train_gen.classes
)
auto_class_weights = dict(enumerate(auto_class_weights))
print("⚖️ Wagi klas:", auto_class_weights)

# ===========================================
# 🧩 KOMPILACJA MODELU Z MAŁYM LEARNING RATE
# ===========================================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),  # delikatny fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ===========================================
# 🛠 CALLBACKS
# ===========================================
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True),
    ModelCheckpoint('models/field_classifier_finetuned.keras', save_best_only=True)
]

# ===========================================
# 🚀 TRENING (FINETUNING)
# ===========================================
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    class_weight=auto_class_weights,
    callbacks=callbacks
)

# ===========================================
# 💾 ZAPIS NOWEGO MODELU
# ===========================================
model.save('models/field_classifier_finetuned.keras')
print("\n💾 Nowy model zapisany jako: models/field_classifier_finetuned.keras")

# ===========================================
# 📈 WYKRES TRENINGU
# ===========================================
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.show()

print("\n✅ Fine-tuning zakończony. Model został zaktualizowany bez przeuczenia.")