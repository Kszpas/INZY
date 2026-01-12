import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
import os


# Ścieżka do przykładowego obrazu z katalogu treningowego
img_path = r'C:\Users\userr\Desktop\INZY\debug_tiles\tile_00.jpg'  # zmień na własny obraz

# 1️⃣ Wczytaj obraz i przeskaluj go do 128x128 pikseli
img = image.load_img(img_path, target_size=(128, 128))

# 2️⃣ Zamień obraz na macierz NumPy
img_array = image.img_to_array(img)

print("Kształt macierzy:", img_array.shape)
print("Przykładowe wartości pikseli (RGB):")
print(img_array[0, 0])  # wartości koloru pierwszego piksela (R, G, B)

# 3️⃣ Przeskaluj wartości (0–1 zamiast 0–255)
img_array = img_array / 255.0

print("\nPo przeskalowaniu:")
print(img_array[0, 0])

# 4️⃣ Wyświetl obraz
plt.imshow(img_array)
plt.title("Obraz wczytany i przeskalowany do 128x128")
plt.axis("off")
plt.show()

# 🔹 Konwersja do macierzy NumPy
img_array = image.img_to_array(img)

print("Kształt macierzy:", img_array.shape)
print("\nPełna macierz wartości RGB (0–255):\n")
print(img_array.astype(int))  # zaokrąglamy do liczb całkowitych dla czytelności