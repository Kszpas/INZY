import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#   Wczytanie gotowego modelu
model = tf.keras.models.load_model('models/field_classifier_final_balanced.keras')

#   Generator dla walidacji (tylko do testu, można dać większy rozmiar)
IMG_SIZE = (128, 128)
val_datagen = ImageDataGenerator(rescale=1./255)

val_gen = val_datagen.flow_from_directory(
    'dataset/val',
    target_size=IMG_SIZE,   # <-- wystarczy zmienić tu, nie w plikach
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

print("\nKolejność klas:", val_gen.class_indices)

#   Predykcje modelu
pred = model.predict(val_gen)
y_pred = np.argmax(pred, axis=1)
y_true = val_gen.classes
class_names = list(val_gen.class_indices.keys())

#   Macierz pomyłek (confusion matrix)
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='Blues',
    xticklabels=class_names, 
    yticklabels=class_names
)
plt.xlabel('🔮 Predicted')
plt.ylabel('✅ True')
plt.title('Macierz Konfuzji')
plt.show()

#   Raport szczegółowy (precision, recall, F1)
print("\n\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))