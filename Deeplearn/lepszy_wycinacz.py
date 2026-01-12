import cv2 as cv
import os
import glob

img = cv.imread("1.png")

board_size = 512
num_squares = 8
square_size = board_size // num_squares

output_dir = "pola"
os.makedirs(output_dir, exist_ok=True)

# 🔹 znajdź najwyższy istniejący numer pliku w folderze
existing_files = glob.glob(os.path.join(output_dir, "*.jpg"))
if existing_files:
    # z nazw typu "pola_00123.jpg" wyciągnij największy numer
    max_num = max([int(os.path.basename(f).split("_")[-1].split(".")[0]) 
                   for f in existing_files if "_" in os.path.basename(f)])
else:
    max_num = 0

counter = max_num + 1  # zacznij od następnego numeru

for row in range(num_squares):
    for col in range(num_squares):
        y1 = row * square_size
        y2 = (row + 1) * square_size
        x1 = col * square_size
        x2 = (col + 1) * square_size

        crop = img[y1:y2, x1:x2]
        filename = os.path.join(output_dir, f"pola_{counter:05d}.jpg")
        cv.imwrite(filename, crop)
        counter += 1

print(f"✅ Zapisano {counter - max_num - 1} nowych plików (rozmiar 64x64) do folderu '{output_dir}'")