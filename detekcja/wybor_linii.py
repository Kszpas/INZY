import cv2 as cv
import numpy as np

# Wczytaj obraz
wczytaj = cv.imread('1000004951.jpg')

height, width = wczytaj.shape[:2]
max_dim = 1024
scale = max_dim / max(height, width)
new_width = int(width * scale)
new_height = int(height * scale)
img = cv.resize(wczytaj, (new_width, new_height), interpolation=cv.INTER_AREA)

clone = img.copy()
points = []

def select_point(event, x, y, flags, param):
    global points
    if event == cv.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv.imshow("Zaznacz rogi", img)

cv.imshow("Zaznacz rogi", img)
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



# Lista punktów
vertical_lines = []
horizontal_lines = []
mode = 'vertical'  # zaczynamy od pionowych

def select_point(event, x, y, flags, param):
    global mode, vertical_lines, horizontal_lines
    if event == cv.EVENT_LBUTTONDOWN:
        if mode == 'vertical':
            vertical_lines.append((x, y))
            cv.circle(warped, (x, y), 4, (0, 255, 0), -1)
            cv.imshow("Zaznacz linie", warped)
        elif mode == 'horizontal':
            horizontal_lines.append((x, y))
            cv.circle(warped, (x, y), 4, (255, 0, 0), -1)
            cv.imshow("Zaznacz linie", warped)

def draw_grid(warped, v_lines, h_lines):
    # Sortujemy punkty według współrzędnych
    v_lines = sorted(v_lines, key=lambda p: p[0])
    h_lines = sorted(h_lines, key=lambda p: p[1])

    grid = warped.copy()
    # Rysujemy pionowe
    for p in v_lines:
        x = p[0]
        cv.line(grid, (x, 0), (x, grid.shape[0]), (0, 0, 255), 2)
    # Rysujemy poziome
    for p in h_lines:
        y = p[1]
        cv.line(grid, (0, y), (grid.shape[1], y), (0, 0, 255), 2)
    return grid


cv.namedWindow("Zaznacz linie", cv.WINDOW_NORMAL)
cv.imshow("Zaznacz linie", warped)
cv.setMouseCallback("Zaznacz linie", select_point)

print("Krok 1️⃣: Klikaj końce PIONOWYCH linii (od lewej do prawej).")
print("❗ Po zakończeniu naciśnij klawisz [v], aby przejść do poziomych linii.")

while True:
    k = cv.waitKey(1) & 0xFF
    if k == ord('v'):
        # przejście do poziomych
        mode = 'horizontal'
        print("Krok 2️⃣: Klikaj końce POZIOMYCH linii (od góry do dołu).")
        print("❗ Po zakończeniu naciśnij [q], aby zakończyć.")
    elif k == ord('q'):
        break

cv.destroyAllWindows()

# Rysujemy ostateczną siatkę
final = draw_grid(warped, vertical_lines, horizontal_lines)
cv.imshow("Siatka szachownicy", final)
cv.imwrite("szachownica_siatka.jpg", final)
cv.waitKey(0)
cv.destroyAllWindows()

print(f"Liczba pionowych linii: {len(vertical_lines)}")
print(f"Liczba poziomych linii: {len(horizontal_lines)}")