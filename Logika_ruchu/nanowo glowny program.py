# --- live_detector_async.py ---
import cv2 as cv
import numpy as np
import chess
import time
import threading
import os
# === PARAMETRY ===
CAMERA_INDEX = 0
OUTPUT_SIZE = (512, 512)
ANALYSIS_INTERVAL = 2.0  # sekundy pomiędzy analizami

# === STAN GLOBALNY ===
latest_state = None
prev_state = None
board = chess.Board()
lock = threading.Lock()
stop_flag = False

# === INICJALIZACJA ===
cap = cv.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Kamera niedostępna.")
    exit()

points, transform_ready, M = [], False, None
vertical_lines, horizontal_lines = [], []
mode = 'none'
warped = None

# --- WYBÓR ROGÓW ---
def select_corner(event, x, y, flags, param):
    global points, transform_ready, M
    if event == cv.EVENT_LBUTTONDOWN and not transform_ready:
        points.append((x, y))
        print(f"Punkt {len(points)}: {x},{y}")
        if len(points) == 4:
            pts1 = np.float32(points)
            pts2 = np.float32([
                [0, 0],
                [OUTPUT_SIZE[0], 0],
                [OUTPUT_SIZE[0], OUTPUT_SIZE[1]],
                [0, OUTPUT_SIZE[1]]
            ])
            M = cv.getPerspectiveTransform(pts1, pts2)
            transform_ready = True
            print("✅ Wybrano 4 punkty — plansza wyprostowana.")

# --- WYBÓR LINII ---
def select_point(event, x, y, flags, param):
    global mode, vertical_lines, horizontal_lines, warped
    if warped is None:
        return
    if event == cv.EVENT_LBUTTONDOWN:
        if mode == 'vertical':
            vertical_lines.append((x, y))
        elif mode == 'horizontal':
            horizontal_lines.append((x, y))

def draw_grid(frame, v_lines, h_lines):
    v_lines = sorted(v_lines, key=lambda p: p[0])
    h_lines = sorted(h_lines, key=lambda p: p[1])
    grid = frame.copy()
    for p in v_lines:
        cv.line(grid, (p[0], 0), (p[0], grid.shape[0]), (0, 0, 255), 2)
    for p in h_lines:
        cv.line(grid, (0, p[1]), (grid.shape[1], p[1]), (0, 0, 255), 2)
    return grid

cv.namedWindow("Kamera")
cv.setMouseCallback("Kamera", select_corner)

# --- FUNKCJE ---
def analyze_board_with_model(warped_img, v_lines, h_lines):
    """Dzielenie szachownicy na 64 pola i zapis każdego pola jako plik JPG (poziomo odwrócone).
       Kamera: widok z góry, białe figury u dołu (a1 po prawej stronie obrazu)."""

    # sortujemy linie: poziome (h) od góry do dołu, pionowe (v) od lewej do prawej
    v = sorted(v_lines, key=lambda p: p[0])
    h = sorted(h_lines, key=lambda p: p[1])

    if len(v) != 9 or len(h) != 9:
        return {}

    tiles, coords = [], []

    # 🔁 iterujemy po wszystkich rzędach od góry do dołu
    for r in range(8):
        y1, y2 = int(h[r][1]), int(h[r + 1][1])

        # 🔁 kolumny od PRAWEJ do LEWEJ (odwrócenie lewo‑prawo)
        for c in range(7, -1, -1):
            x1, x2 = int(v[c][0]), int(v[c + 1][0])

            # wycinamy odpowiedni fragment planszy
            tile = warped_img[y1:y2, x1:x2]
            tile = cv.cvtColor(tile, cv.COLOR_BGR2RGB)
            tile = cv.resize(tile, (128, 128)).astype(np.float32) / 255.0

            # nazwa pola: kolumny idą odwrotnie (h→a), rzędy jak wcześniej (1–8)
            file_char = chr(ord('a') + (7 - c))  # odwrócenie kolumn
            rank = r + 1
            sq = f"{file_char}{rank}"

            tiles.append(tile)
            coords.append(sq)

    print(f"📷 Wyodrębniono {len(tiles)} pól szachownicy (odwrócone lewo‑prawo).")

    # --- ZAPIS CAŁEJ PLANSZY ---
    save_all_dir = "all_tiles"
    os.makedirs(save_all_dir, exist_ok=True)

    for tile_debug, sq in zip(tiles, coords):
        out = (tile_debug * 255).astype(np.uint8)
        out = cv.cvtColor(out, cv.COLOR_RGB2BGR)
        filename = f"{sq}.jpg"
        cv.imwrite(os.path.join(save_all_dir, filename), out)

    print("💾 Zapisano całą szachownicę (64 pola) do folderu 'all_tiles' (odwrócenie lewo‑prawo).")

    return coords

# --- OBSŁUGA WĄTKU PREDYKCJI (teraz tylko geometria co 2 sekundy) ---
def prediction_loop():
    global stop_flag
    print("▶️ Asynchroniczna analiza co 2 sekundy uruchomiona.")
    while not stop_flag:
        time.sleep(ANALYSIS_INTERVAL)
        if not transform_ready or len(vertical_lines) != 9 or len(horizontal_lines) != 9:
            continue
        with lock:
            img_copy = None if warped is None else warped.copy()
        if img_copy is None:
            continue
        state = analyze_board_with_model(img_copy, vertical_lines, horizontal_lines)
        if not state:
            continue
        print("INFO: analiza wykonana, przykładowe pola:", state[:8])

# --- GŁÓWNA PĘTLA KAMERY ---
print("Sterowanie: [v] - pionowe, [h] - poziome, [q] - start")

predict_thread = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    for p in points:
        cv.circle(frame, p, 5, (0, 255, 0), -1)
    if transform_ready and M is not None:
        warped = cv.warpPerspective(frame, M, OUTPUT_SIZE)
        if len(vertical_lines) > 0 or len(horizontal_lines) > 0:
            grid = draw_grid(warped, vertical_lines, horizontal_lines)
            cv.imshow("Zaznacz linie", grid)
        else:
            cv.imshow("Zaznacz linie", warped)
    cv.imshow("Kamera", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('v'):
        mode = 'vertical'
        cv.setMouseCallback("Zaznacz linie", select_point)
        print("🟩 Klikaj 9 pionowych linii.")
    elif key == ord('h'):
        mode = 'horizontal'
        cv.setMouseCallback("Zaznacz linie", select_point)
        print("🟦 Klikaj 9 poziomych linii.")
    elif key == ord('q'):
        print("▶️ Start analizy w tle...")
        if predict_thread is None:
            predict_thread = threading.Thread(target=prediction_loop, daemon=True)
            predict_thread.start()
        break

# --- PETLA ŻYWEGO PODGLĄDU ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    warped = cv.warpPerspective(frame, M, OUTPUT_SIZE)
    with lock:
        copy_for_thread = warped.copy()
    cv.imshow("Podglad", copy_for_thread)
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        stop_flag = True
        break

cap.release()
cv.destroyAllWindows()
print("🛑 Zatrzymano analizę.")