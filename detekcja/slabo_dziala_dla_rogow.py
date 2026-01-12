import cv2 as cv
import numpy as np

# Wczytaj obraz
img = cv.imread('1000004951.jpg')

c

clone = rescaled_image.copy()
points = []

def select_point(event, x, y, flags, param):
    global points
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv.circle(rescaled_image, (x, y), 5, (0, 255, 0), -1)
        cv.imshow("Zaznacz rogi", rescaled_image)

cv.imshow("Zaznacz rogi", rescaled_image)
cv.setMouseCallback("Zaznacz rogi", select_point)

print("Kliknij kolejno 4 rogi planszy: lewy-górny, prawy-górny, prawy-dolny, lewy-dolny...")
cv.waitKey(0)
cv.destroyAllWindows()

print("Zaznaczone punkty:", points)

# Sprawdź, czy są 4 punkty i przekształć perspektywę
if len(points) == 4:
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [512, 0], [512, 512], [0, 512]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    warped = cv.warpPerspective(clone, M, (512, 512))
    cv.imshow('Szachownica wyprostowana', warped)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite('szachownica_wyprostowana.jpg', warped)
else:
    print("Musisz zaznaczyć dokładnie 4 punkty!")
    
    
    
# Zamień na skalę szarości (zmniejsza szum i ułatwia detekcję)
gray = cv.cvtColor(warped, cv.COLOR_BGR2GRAY)

# Opcjonalnie: odszumienie przez rozmycie
blur = cv.GaussianBlur(gray, (5,5), 0)

# Wykrywanie krawędzi metodą Canny
edges = cv.Canny(blur, 50, 150)
cv.imshow('Krawędzie (Canny)', edges)

# Wykrywanie linii metodą Hougha
lines = cv.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)

# Narysuj linie na kopii obrazu
line_img = warped.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(line_img, (x1, y1), (x2, y2), (0,255,0), 2)

cv.imshow('Krawędzie szachownicy', line_img)

cv.waitKey(0)
cv.destroyAllWindows()