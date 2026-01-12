import cv2 as cv
import numpy as np

# Wybierz źródło (dla OBS często 1 lub 2)
cap = cv.VideoCapture(0)



points = []
transform_ready = False   # flaga: czy wybrano 4 punkty
M = None                  # macierz transformacji
output_size = (512, 512)  # rozmiar prostokątnej szachownicy

def select_point(event, x, y, flags, param):
    global points, transform_ready, M

    if event == cv.EVENT_LBUTTONDOWN and not transform_ready:
        points.append((x, y))
        print(f"Punkt {len(points)}: {x}, {y}")

        if len(points) == 4:
            print("✅ Wybrano 4 punkty – transformacja gotowa.")
            pts1 = np.float32(points)
            pts2 = np.float32([[0, 0], [output_size[0], 0],
                               [output_size[0], output_size[1]], [0, output_size[1]]])
            M = cv.getPerspectiveTransform(pts1, pts2)
            transform_ready = True

cv.namedWindow("Kamera")
cv.setMouseCallback("Kamera", select_point)

print("📸 Kliknij 4 rogi planszy: lewy-górny, prawy-górny, prawy-dolny, lewy-dolny")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rysowanie wybranych punktów na podglądzie
    for p in points:
        cv.circle(frame, p, 5, (0, 255, 0), -1)
    if len(points) == 4:
        cv.polylines(frame, [np.int32(points)], True, (0, 255, 255), 2)

    # Jeśli transformacja gotowa → przekształć bieżącą klatkę
    if transform_ready and M is not None:
        warped = cv.warpPerspective(frame, M, output_size)
        cv.imshow("Wyprostowana szachownica", warped)

    # Wyświetl podgląd z kamerki
    cv.imshow("Kamera", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('r'):  # resetuj punkty
        points = []
        transform_ready = False
        print("🔁 Zresetowano punkty, wybierz ponownie.")
    elif key == ord('q'):
        break


# ==========================================
# FRAGMENT „ZAZNACZ LINIE” – OBRAZ NA ŻYWO
# ==========================================

vertical_lines = []
horizontal_lines = []
mode = 'vertical'  # zaczynamy od pionowych

def select_point(event, x, y, flags, param):
    global mode, vertical_lines, horizontal_lines, warped
    if event == cv.EVENT_LBUTTONDOWN:
        if mode == 'vertical':
            vertical_lines.append((x, y))
        elif mode == 'horizontal':
            horizontal_lines.append((x, y))

def draw_grid(frame, v_lines, h_lines):
    # Sortujemy punkty
    v_lines = sorted(v_lines, key=lambda p: p[0])
    h_lines = sorted(h_lines, key=lambda p: p[1])
    grid = frame.copy()
    for p in v_lines:
        cv.line(grid, (p[0], 0), (p[0], grid.shape[0]), (0, 0, 255), 2)
    for p in h_lines:
        cv.line(grid, (0, p[1]), (grid.shape[1], p[1]), (0, 0, 255), 2)
    return grid

cv.namedWindow("Zaznacz linie", cv.WINDOW_NORMAL)
cv.setMouseCallback("Zaznacz linie", select_point)

print("Krok 1️⃣: Klikaj końce PIONOWYCH linii (od lewej do prawej).")
print("❗ Po zakończeniu naciśnij [v], aby przejść do poziomych.")
print("❗ Naciśnij [q], aby zakończyć.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # przekształcenie bieżącej klatki (perspektywa wyprostowana)
    warped = cv.warpPerspective(frame, M, (512, 512))

    # podgląd z zaznaczaniem linii w czasie rzeczywistym
    preview = draw_grid(warped, vertical_lines, horizontal_lines)
    cv.imshow("Zaznacz linie", preview)

    k = cv.waitKey(1) & 0xFF
    if k == ord('v'):
        mode = 'horizontal'
        print("Krok 2️⃣: Klikaj końce POZIOMYCH linii (od góry do dołu).")
    elif k == ord('q'):
        break

cv.destroyAllWindows()

print(f"Pionowe linie: {len(vertical_lines)}")
print(f"Poziome linie: {len(horizontal_lines)}")

cap.release()
cv.destroyAllWindows()
