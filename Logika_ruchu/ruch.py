# --- live_detector.py ---
import cv2 as cv
import numpy as np
import time
import tensorflow as tf
import chess

# === PARAMETRY ===
CAMERA_INDEX = 0
OUTPUT_SIZE = (512, 512)
MODEL_PATH = "models/field_classifier_final_balanced.keras"
CLASS_NAMES = ["empty", "white", "black"]
ANALYSIS_INTERVAL = 2.0  # sekundy

# === INICJALIZACJA ===
model = tf.keras.models.load_model(MODEL_PATH)
cap = cv.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Kamera niedostępna.")
    exit()

print("Kliknij 4 rogi planszy (lewy‑górny, prawy‑górny, prawy‑dolny, lewy‑dolny).")
points, transform_ready, M = [], False, None
vertical_lines, horizontal_lines = [], []
mode = 'none'
warped = None

# --- wybór rogów ---
def select_corner(event, x, y, flags, param):
    global points, transform_ready, M
    if event == cv.EVENT_LBUTTONDOWN and not transform_ready:
        points.append((x, y))
        print(f"Punkt {len(points)}: {x},{y}")
        if len(points) == 4:
            pts1 = np.float32(points)
            pts2 = np.float32([
                [0, 0],
                [OUTPUT_SIZE[0], 0],
                [OUTPUT_SIZE[0], OUTPUT_SIZE[1]],
                [0, OUTPUT_SIZE[1]],
            ])
            M = cv.getPerspectiveTransform(pts1, pts2)
            transform_ready = True
            print("✅ Wybrano 4 punkty – plansza wyprostowana.")

# --- wybór linii ---
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
    v, h = sorted(v_lines, key=lambda p: p[0]), sorted(h_lines, key=lambda p: p[1])
    g = frame.copy()
    for p in v: cv.line(g, (p[0], 0), (p[0], g.shape[0]), (0, 0, 255), 2)
    for p in h: cv.line(g, (0, p[1]), (g.shape[1], p[1]), (0, 0, 255), 2)
    return g

cv.namedWindow("Kamera")
cv.setMouseCallback("Kamera", select_corner)

# --- funkcje pomocnicze ---
def analyze_board_with_model(warped_img, v_lines, h_lines):
    v = sorted(v_lines, key=lambda p: p[0])
    h = sorted(h_lines, key=lambda p: p[1])
    if len(v) != 9 or len(h) != 9:
        print("⚠️ Potrzebne 9 linii pionowych i 9 poziomych.")
        return {}
    fields, tiles, coords = {}, [], []
    for r in range(8):
        y1, y2 = int(h[r][1]), int(h[r+1][1])
        for c in range(8):
            x1, x2 = int(v[c][0]), int(v[c+1][0])
            tile = warped_img[y1:y2, x1:x2]
            tile = cv.resize(tile, (128, 128 )).astype(np.float32)/255.0
            tiles.append(tile)
            coords.append((r, c))
    preds = model.predict(np.stack(tiles), verbose=0)
    labels = [CLASS_NAMES[np.argmax(p)] for p in preds]
    for (r, c), lbl in zip(coords, labels):
        sq = f"{chr(ord('a') + c)}{8 - r}"
        fields[sq] = lbl
    return fields

def detect_move(before, after):
    from_sq, to_sq = None, None
    for sq in before:
        if before[sq] != after[sq]:
            if after[sq] == "empty" and before[sq] != "empty":
                from_sq = sq
            elif before[sq] == "empty" and after[sq] != "empty":
                to_sq = sq
    return f"{from_sq}{to_sq}" if from_sq and to_sq else None

# --- definicja początkowa ---
board = chess.Board()
prev_state = None

print("Sterowanie: [v] – linie pionowe, [h] – poziome, [q] – start")

# --- konfiguracja planszy ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    for p in points:
        cv.circle(frame, p, 5, (0, 255, 0), -1)
    if transform_ready:
        warped = cv.warpPerspective(frame, M, OUTPUT_SIZE)
        if mode != 'none':
            grid = draw_grid(warped, vertical_lines, horizontal_lines)
            cv.imshow("Zaznacz linie", grid)
        else:
            cv.imshow("Zaznacz linie", warped)
    cv.imshow("Kamera", frame)
    key = cv.waitKey(1) & 0xFF
    if key == ord('v'):
        mode = 'vertical'
        cv.setMouseCallback("Zaznacz linie", select_point)
        print("🟩 Klikaj 9 pionowych linii.")
    elif key == ord('h'):
        mode = 'horizontal'
        cv.setMouseCallback("Zaznacz linie", select_point)
        print("🟦 Klikaj 9 poziomych linii.")
    elif key == ord('q'):
        print("▶️ Start analizy co 2 sekundy...")
        break

# --- analiza w czasie rzeczywistym ---
while True:
    ret, frame = cap.read()
    if not ret:
        break
    warped = cv.warpPerspective(frame, M, OUTPUT_SIZE)
    state = analyze_board_with_model(warped, vertical_lines, horizontal_lines)
    if not state:
        continue
    if prev_state is not None:
        move = detect_move(prev_state, state)
        if move:
            mv = chess.Move.from_uci(move)
            if mv in board.legal_moves:
                board.push(mv)
                print("♟️ Ruch wykonany:", mv)
                print(board)
    prev_state = state
    time.sleep(ANALYSIS_INTERVAL)

cap.release()
cv.destroyAllWindows()