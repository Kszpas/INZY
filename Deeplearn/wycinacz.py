import cv2 as cv
import os

img = cv.imread("3.jpg")

# --- parametry ---
board_size = 512     # szerokość/wysokość całego obrazu
num_squares = 8      # 8x8
square_size = board_size // num_squares  # 64 piksele

# --- upewnij się, że obraz jest 512x512 ---
img = cv.resize(img, (board_size, board_size))

# --- folder do zapisu ---
os.makedirs("pola", exist_ok=True)

counter = 0
for row in range(num_squares):
    for col in range(num_squares):
        # współrzędne pola (y1:y2 , x1:x2)
        y1 = row * square_size
        y2 = (row + 1) * square_size
        x1 = col * square_size
        x2 = (col + 1) * square_size

        crop = img[y1:y2, x1:x2]
        cv.imwrite(f"pola/{row}_{col}.jpg", crop)
        counter += 1

print(f"Zapisano {counter} plików 64x64 do folderu 'pola/'")