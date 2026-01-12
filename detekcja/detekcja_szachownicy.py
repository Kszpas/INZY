import cv2 as cv
import numpy as np

img = cv.imread('1000004951.jpg')
# cv.imshow("Szachy", img)

# Przeskalowanie do rozmiaru 1024x1024 zachowując proporcje
height, width = img.shape[:2]
max_dim = 1024
scale = max_dim / max(height, width)
new_width = int(width * scale)
new_height = int(height * scale)
rescaled_image = cv.resize(img, (new_width, new_height), interpolation=cv.INTER_AREA)
cv.imshow("Rescaled Image", rescaled_image)

blank = np.zeros(rescaled_image.shape[:2], dtype='uint8')
# Wyświetlenie pustego obrazu dla debugowania
#cv.imshow('Blank Image', blank)

# Obliczenie współrzędnych prostokąta proporcjonalnie do rozmiaru obrazu
rect_start_x = int(new_width * 0.05)  # 10% od lewej krawędzi
rect_start_y = int(new_height * 0.15)  # 10% od górnej krawędzi
rect_end_x = int(new_width * 0.95)    # 90% szerokości
rect_end_y = int(new_height * 0.8)    # 90% wysokości

# Rysowanie prostokąta
maska = cv.rectangle(blank.copy(), 
                        (rect_start_x, rect_start_y), 
                        (rect_end_x, rect_end_y),  
                        255,  # kolor (biały)
                        -1)    # grubość linii (2 piksele)

# Wyświetlenie obrazu z prostokątem
#cv.imshow('Rectangle on Blank', maska)

zamaskowany = cv.bitwise_and(rescaled_image, rescaled_image, mask=maska)
#cv.imshow('Zamaskowany Obraz', zamaskowany)



# Zamień na skalę szarości (zmniejsza szum i ułatwia detekcję)
gray = cv.cvtColor(zamaskowany, cv.COLOR_BGR2GRAY)

# Opcjonalnie: odszumienie przez rozmycie
blur = cv.GaussianBlur(gray, (5,5), 0)

# Wykrywanie krawędzi metodą Canny
edges = cv.Canny(blur, 50, 150)
cv.imshow('Krawędzie (Canny)', edges)

# Wykrywanie linii metodą Hougha
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Narysuj linie na kopii obrazu
line_img = zamaskowany.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_img, (x1, y1), (x2, y2), (0,255,0), 2)

cv.imshow('Krawędzie szachownicy', line_img)


cv.waitKey(0)
cv.destroyAllWindows()
