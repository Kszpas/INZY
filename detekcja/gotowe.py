import cv2 as cv
import numpy as np

# --- PARAMETRY ---
CAMERA_INDEX = 0          # ustaw numer kamery (np. 0, 1, 2 dla OBS)
OUTPUT_SIZE = (512, 512)  # rozmiar wyprostowanego widoku

# --- INICJALIZACJA ---
cap = cv.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Nie udało się otworzyć kamery.")
    exit()

points = []
transform_ready = False
M = None
warped = None

# --- Wybór 4 punktów --- #
def select_corner(event, x, y, flags, param):
    global points, transform_ready, M
    if event == cv.EVENT_LBUTTONDOWN and not transform_ready:
        points.append((x, y))
        print(f"Punkt {len(points)}: {x},{y}")
        if len(points) == 4:
            pts1 = np.float32(points)
            pts2 = np.float32([[0, 0], [OUTPUT_SIZE[0], 0],
                               [OUTPUT_SIZE[0], OUTPUT_SIZE[1]], [0, OUTPUT_SIZE[1]]])
            M = cv.getPerspectiveTransform(pts1, pts2)
            transform_ready = True
            print(" Wybrano 4 punkty.")

cv.namedWindow("Kamera")
cv.setMouseCallback("Kamera", select_corner)

# --- Zmienne do rysowania siatki ---
vertical_lines = []
horizontal_lines = []
mode = 'none'   # 'none', 'vertical', 'horizontal'

def select_point(event, x, y, flags, param):
    global mode, vertical_lines, horizontal_lines, warped
    if warped is None:
        return
    if event == cv.EVENT_LBUTTONDOWN:
        if mode == 'vertical':
            vertical_lines.append((x, y))
            cv.circle(warped, (x, y), 4, (0, 255, 0), -1)
        elif mode == 'horizontal':
            horizontal_lines.append((x, y))
            cv.circle(warped, (x, y), 4, (255, 0, 0), -1)

def draw_grid(frame, v_lines, h_lines):
    v_lines = sorted(v_lines, key=lambda p: p[0])
    h_lines = sorted(h_lines, key=lambda p: p[1])
    grid = frame.copy()
    for p in v_lines:
        x = p[0]
        cv.line(grid, (x, 0), (x, grid.shape[0]), (0, 0, 255), 2)
    for p in h_lines:
        y = p[1]
        cv.line(grid, (0, y), (grid.shape[1], y), (0, 0, 255), 2)
    return grid

cv.setMouseCallback("Kamera", select_corner)  # najpierw tryb rogów

print(" Kliknij 4 rogi planszy (kolejność: lewy-górny, prawy-górny, prawy-dolny, lewy-dolny).")

# --- Główna pętla ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rysowanie wybranych rogów
    for p in points:
        cv.circle(frame, p, 5, (0, 255, 0), -1)

    # Jeśli mamy 4 punkty → przekształcenie
    if transform_ready and M is not None:
        warped = cv.warpPerspective(frame, M, OUTPUT_SIZE)
        cv.imwrite('szachownica_wyprostowana.jpg', warped)

        # wybór linii dopiero po przekształceniu
        if mode == 'none':
            cv.namedWindow("Zaznacz linie", cv.WINDOW_NORMAL)
            cv.setMouseCallback("Zaznacz linie", select_point)
            print("Klikaj końce PIONOWYCH linii (od lewej do prawej).")
            print(" Po zakończeniu naciśnij [v], aby przejść do poziomych.")
            mode = 'vertical'

        # Rysowanie klikniętych linii
        if len(vertical_lines) > 0 or len(horizontal_lines) > 0:
            grid = draw_grid(warped, vertical_lines, horizontal_lines)
            cv.imshow("Zaznacz linie", grid)
        else:
            cv.imshow("Zaznacz linie", warped)

    cv.imshow("Kamera", frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('v') and mode == 'vertical':
        mode = 'horizontal'
        print(" Klikaj końce POZIOMYCH linii (od góry do dołu).")
        print(" Naciśnij [q], aby zakończyć.")
    elif key == ord('r'):
        points = []
        vertical_lines = []
        horizontal_lines = []
        transform_ready = False
        mode = 'none'
        print("🔁 Reset – wybierz punkty od nowa.")
    elif key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

print(f"Pionowe linie: {len(vertical_lines)} | Poziome linie: {len(horizontal_lines)}")