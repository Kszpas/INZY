import cv2 as cv
import numpy as np
import chess
import time
import threading
import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
from tkinter import font
import plansza

# === PARAMETRY ===
CAMERA_INDEX = 1
OUTPUT_SIZE = (512, 512)
ANALYSIS_INTERVAL = 3  # sekundy między analizami
CLASS_NAMES = ["black", "empty", "white"]
REQUIRED_CONSECUTIVE = 2  # ile kolejnych wykryć tego samego ruchu wymagać, by potwierdzić

# === MODEL ===
model = tf.keras.models.load_model('models/field_classifier_finetuned.keras')

# === STAN GLOBALNY ===
latest_state = None
prev_state = None
board = chess.Board()
lock = threading.Lock()
stop_flag = False
move_history = []
MOVE_HISTORY_FILE = os.path.join(os.path.dirname(__file__), 'move_history.txt')
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), 'camera_calibration.json')

# === INICJALIZACJA ===
cap = cv.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print("❌ Kamera niedostępna.")
    exit()

points, transform_ready, M = [], False, None
vertical_lines, horizontal_lines = [], []
mode = 'none'
warped = None
# tkinter root (will be created when entering preview mode)
root = None
analysis_counter = 0


def compute_perspective_matrix(corner_points):
    """Build perspective transform matrix from 4 clicked corners."""
    if len(corner_points) != 4:
        return None
    pts1 = np.float32(corner_points)
    pts2 = np.float32([
        [0, 0],
        [OUTPUT_SIZE[0], 0],
        [OUTPUT_SIZE[0], OUTPUT_SIZE[1]],
        [0, OUTPUT_SIZE[1]]
    ])
    return cv.getPerspectiveTransform(pts1, pts2)


def save_calibration():
    """Persist selected corners and grid lines so calibration survives restart."""
    data = {
        "camera_index": CAMERA_INDEX,
        "output_size": list(OUTPUT_SIZE),
        "points": [list(p) for p in points],
        "vertical_lines": [list(p) for p in vertical_lines],
        "horizontal_lines": [list(p) for p in horizontal_lines],
    }
    try:
        with open(CALIBRATION_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Nie mozna zapisac kalibracji: {e}")


def load_calibration():
    """Load saved calibration from disk if available and valid."""
    global points, vertical_lines, horizontal_lines, transform_ready, M
    if not os.path.exists(CALIBRATION_FILE):
        return False

    try:
        with open(CALIBRATION_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Nie mozna wczytac kalibracji: {e}")
        return False

    loaded_points = [tuple(p) for p in data.get("points", [])]
    loaded_vertical = [tuple(p) for p in data.get("vertical_lines", [])]
    loaded_horizontal = [tuple(p) for p in data.get("horizontal_lines", [])]

    if len(loaded_points) != 4:
        print("Pomijam kalibracje: brak 4 punktow rogow.")
        return False

    matrix = compute_perspective_matrix(loaded_points)
    if matrix is None:
        return False

    points = loaded_points
    vertical_lines = loaded_vertical[:9]
    horizontal_lines = loaded_horizontal[:9]
    M = matrix
    transform_ready = True

    print(
        "Zaladowano kalibracje: "
        f"rogi=4, pionowe={len(vertical_lines)}, poziome={len(horizontal_lines)}"
    )
    return True


def reset_calibration(remove_saved=True):
    """Clear in-memory calibration and optionally remove saved file."""
    global points, transform_ready, M, vertical_lines, horizontal_lines, mode
    points = []
    transform_ready = False
    M = None
    vertical_lines = []
    horizontal_lines = []
    mode = 'none'
    if remove_saved and os.path.exists(CALIBRATION_FILE):
        try:
            os.remove(CALIBRATION_FILE)
            print("Usunieto zapis kalibracji.")
        except Exception as e:
            print(f"Nie mozna usunac pliku kalibracji: {e}")


# --- WYBÓR ROGÓW ---
def select_corner(event, x, y, flags, param):
    global points, transform_ready, M
    if event == cv.EVENT_LBUTTONDOWN and not transform_ready:
        points.append((x, y))
        print(f"Punkt {len(points)}: {x},{y}")
        if len(points) == 4:
            M = compute_perspective_matrix(points)
            transform_ready = True
            print("✅ Wybrano 4 punkty — plansza wyprostowana.")
            save_calibration()

# --- WYBÓR LINII ---
def select_point(event, x, y, flags, param):
    global mode, vertical_lines, horizontal_lines, warped
    if warped is None:
        return
    if event == cv.EVENT_LBUTTONDOWN:
        if mode == 'vertical':
            if len(vertical_lines) < 9:
                vertical_lines.append((x, y))
                print(f"Pionowe: {len(vertical_lines)}/9")
                save_calibration()
            else:
                print("Masz juz 9 linii pionowych.")
        elif mode == 'horizontal':
            if len(horizontal_lines) < 9:
                horizontal_lines.append((x, y))
                print(f"Poziome: {len(horizontal_lines)}/9")
                save_calibration()
            else:
                print("Masz juz 9 linii poziomych.")

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
load_calibration()

# --- ANALIZA PLANSZY ---
def analyze_board_with_model(warped_img, v_lines, h_lines):
    global analysis_counter
    v = sorted(v_lines, key=lambda p: p[0])
    h = sorted(h_lines, key=lambda p: p[1])
    if len(v) != 9 or len(h) != 9:
        return {}

    tiles, coords = [], []

    # Segmentacja na 64 poszczególne pola
    for r in range(8):
        y1, y2 = int(h[r][1]), int(h[r + 1][1])
        for c in range(8):  # od lewej do prawej
            x1, x2 = int(v[c][0]), int(v[c + 1][0])
            tile = warped_img[y1:y2, x1:x2]
            tile = cv.cvtColor(tile, cv.COLOR_BGR2RGB)
            tile = cv.resize(tile, (128, 128)).astype(np.float32) / 255.0

            file_char = chr(ord('a') + c)
            rank = 8 - r
            sq = f"{file_char}{rank}"

            tiles.append(tile)
            coords.append(sq)

    preds = model.predict(np.stack(tiles), verbose=0)
    labels = [CLASS_NAMES[np.argmax(p)] for p in preds]
    board_results = dict(zip(coords, labels))

    # Czytelny wydruk: każda analiza ma swój blok z nagłówkiem i separatorami.
    analysis_counter += 1
    stamp = datetime.now().strftime('%H:%M:%S')
    symbol_map = {"white": "W", "black": "B", "empty": "."}
    print("\n" + "=" * 78)
    print(f"ANALIZA #{analysis_counter} | {stamp} | klasy: W=white, B=black, .=empty")
    print("      a  b  c  d  e  f  g  h")

    for rank in range(8, 0, -1):
        rank_squares = [f"{chr(ord('a') + c)}{rank}" for c in range(8)]
        rank_data = [(sq, board_results[sq]) for sq in rank_squares]
        row_symbols = "  ".join(symbol_map[board_results[sq]] for sq in rank_squares)
        print(f"Rzad {rank}: {row_symbols}   {rank_data}")

    print("=" * 78 + "\n")

    return board_results


def save_move_to_file(text):
    try:
        with open(MOVE_HISTORY_FILE, 'a', encoding='utf-8') as f:
            f.write(text + "\n")
    except Exception as e:
        print(f"Nie można zapisać historii ruchów: {e}")


def try_apply_move(fr, to):
    """Próbuje zastosować ruch na `board`. Zwraca (applied:bool, desc:str).
    Próbuje UCI, a jeżeli to nielegalne, próbuje wariantów promocji (q,r,b,n).
    """
    global board

    if fr == '?' or to == '?':
        return False, f"Niepełny ruch: {fr} → {to}"

    uci = f"{fr}{to}"
    # najpierw spróbuj zwykły ruch
    try:
        mv = chess.Move.from_uci(uci)
    except Exception:
        mv = None

    # jeśli istnieje i jest prawny, zastosuj
    if mv is not None and mv in board.legal_moves:
        san = board.san(mv)
        board.push(mv)
        return True, f"{uci} {san}"

    # spróbuj promocji (jeżeli pionek doszedł na ostatnią linię)
    for p in ['q', 'r', 'b', 'n']:
        try:
            mvp = chess.Move.from_uci(uci + p)
        except Exception:
            mvp = None
        if mvp is not None and mvp in board.legal_moves:
            san = board.san(mvp)
            board.push(mvp)
            return True, f"{uci}{p} {san}"

    return False, f"Nielegalny lub nieznany ruch: {uci}"


def draw_move_history(img, history, max_lines=6):
    """Rysuje krótką historię ruchów na obrazie (używane w podglądzie)."""
    h = img.copy()
    x, y = 8, 20
    font = cv.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    color = (255, 255, 255)
    thickness = 1
    bg_color = (0, 0, 0)
    lines = history[-max_lines:]
    # draw semi-transparent background
    overlay = h.copy()
    cv.rectangle(overlay, (x - 4, y - 14), (220, y + 16 * len(lines)), bg_color, -1)
    alpha = 0.5
    cv.addWeighted(overlay, alpha, h, 1 - alpha, 0, h)
    for i, ln in enumerate(reversed(lines)):
        txt = ln
        cv.putText(h, txt, (x, y + i * 16), font, scale, color, thickness, cv.LINE_AA)
    return h

# --- 🆕 FUNKCJA WYKRYWANIA RUCHU ---
def detect_move(prev_state, current_state):
    """
    Wykrywa ruch na podstawie różnicy pomiędzy dwoma stanami predykcji.

    Zwraca:
      - None jeżeli brak rozróżnialnej zmiany,
      - krotkę (from_sq, to_sq, color) gdzie from_sq lub to_sq mogą być '?' gdy
        nie da się jednoznacznie określić źródła/celu (np. pojawienie/zanik).

    Uwaga: funkcja jedynie porównuje dwa stany i zwraca kandydatów na ruch.
    Decyzja o potwierdzeniu wymaga kilku kolejnych wywołań (zrobione w pętli).
    """
    if prev_state is None or current_state is None:
        return None

    changes = [sq for sq in current_state if prev_state.get(sq) != current_state[sq]]
    if not changes:
        return None

    disappeared = [sq for sq in changes if current_state[sq] == "empty" and prev_state[sq] != "empty"]
    appeared = [sq for sq in changes if current_state[sq] != "empty" and prev_state[sq] == "empty"]

    # najbardziej typowy ruch: jedno pole zniknęło, jedno pole pojawiło się
    if len(appeared) == 1 and len(disappeared) == 1:
        from_sq = disappeared[0]
        to_sq = appeared[0]
        color = current_state[to_sq]
        return (from_sq, to_sq, color)

    # tylko pojawienie — nie znamy źródła
    if len(appeared) == 1:
        to_sq = appeared[0]
        color = current_state[to_sq]
        return ('?', to_sq, color)

    # tylko zniknięcie — nie znamy celu
    if len(disappeared) == 1:
        from_sq = disappeared[0]
        color = prev_state[from_sq]
        return (from_sq, '?', color)

    # złożone zmiany — nie potrafimy zidentyfikować pojedynczego ruchu
    return None

# --- OBSŁUGA WĄTKU PREDYKCJI ---
def prediction_loop():
    global stop_flag, prev_state, candidate_move
    print("▶️ Analiza co 5 sekund uruchomiona.")

    # Stan bazowy i licznik do potwierdzania ruchu
    prev_state = None
    candidate_move = None
    candidate_count = 0

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
        # Jeżeli nie mamy jeszcze stanu odniesienia, ustaw go i czekaj na kolejną iterację
        if prev_state is None:
            prev_state = state
            candidate_move = None
            candidate_count = 0
            continue

        move = detect_move(prev_state, state)

        if move is None:
            # brak wykrytego ruchu -> zresetuj licznik
            candidate_move = None
            candidate_count = 0
            continue

        # Jeżeli wykryto ruch, porównaj z poprzednim kandydatem
        if move == candidate_move:
            candidate_count += 1
        else:
            candidate_move = move
            candidate_count = 1

        # Potwierdź ruch dopiero po wymaganej liczbie kolejnych detekcji
        if candidate_count >= REQUIRED_CONSECUTIVE:
            fr, to, color = candidate_move
            frs = fr if fr is not None else '?'
            tos = to if to is not None else '?'
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Spróbuj zaktualizować obiekt chess.Board()
            with lock:
                applied, desc = try_apply_move(fr, to)
                if applied:
                    log_line = f"{timestamp}  APPLIED  {desc}"
                    move_history.append(log_line)
                    save_move_to_file(log_line)
                    print(f"♟️ Pewny ruch i zastosowano: {desc}")
                else:
                    # If a detected move couldn't be applied, do not persist it.
                    print(f"⚠️ Potwierdzono detekcję, ale nie zastosowano: {desc}")

            # zresetuj i ustaw nowy stan odniesienia
            candidate_move = None
            candidate_count = 0
            prev_state = state
        
# --- GŁÓWNA PĘTLA ---
print("Sterowanie: [v] - pionowe, [h] - poziome, [r] - reset kalibracji, [q] - start")

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
    elif key == ord('r'):
        reset_calibration(remove_saved=True)
        print("Reset kalibracji. Wybierz 4 rogi od nowa.")
    elif key == ord('q'):
        print("▶️ Start analizy w tle...")
        if predict_thread is None:
            predict_thread = threading.Thread(target=prediction_loop, daemon=True)
            predict_thread.start()
        # zamiast wchodzić w blokujący loop OpenCV w tym wątku, uruchomimy
        # pętlę podglądu w tle (wątek) i tkinter w wątku głównym — to pozwala
        # na współdzielenie zasobów (kamera i model) w jednym procesie.
        break

def preview_loop(root_ref):
    """Pętla podglądu uruchamiana w tle — pokazuje okno OpenCV tak jak wcześniej.

    Jeżeli użytkownik naciśnie 'q' w tym oknie, ustawi `stop_flag` i poprosi
    tkinter o zakończenie (root.quit via after).
    """
    global warped, stop_flag
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for p in points:
            cv.circle(frame, p, 5, (0, 255, 0), -1)
        if not transform_ready or M is None:
            hint = frame.copy()
            cv.putText(hint, 'Wybierz 4 rogi planszy w oknie "Kamera" (kliknij)', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)
            cv.imshow("Podglad", hint)
        else:
            warped = cv.warpPerspective(frame, M, OUTPUT_SIZE)
            with lock:
                copy_for_thread = warped.copy()
                if move_history:
                    display = draw_move_history(copy_for_thread, move_history)
                else:
                    display = copy_for_thread
            cv.imshow("Podglad", display)
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            stop_flag = True
            # poproś tkinter o zakończenie głównej pętli
            try:
                root_ref.after(0, root_ref.quit)
            except Exception:
                pass
            break

    cap.release()
    cv.destroyAllWindows()
    print("🛑 Zatrzymano analizę.")


# --- Uruchomienie tkinter i wątku podglądu w tym samym procesie ---
def start_tkinter_board_view(square_size: int = 128, poll_ms: int = 500):
    """Utwórz okno tkinter z planszą i aktualizuj ją z globalnego `board`.

    Ta funkcja blokuje (root.mainloop) i powinna być wywołana w wątku głównym.
    """
    global root
    root = tk.Tk()
    root.title('Wizualizacja planszy (w tym samym procesie)')
    cols = 8
    rows = 8
    width = cols * square_size
    height = rows * square_size
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack()
    piece_font = font.Font(family='Segoe UI Symbol', size=max(10, int(square_size * 0.6)))

    def tk_update():
        try:
            mat = plansza.board_to_matrix(board)
            plansza._draw_matrix_on_canvas(canvas, mat, square_size, piece_font)
        except Exception as e:
            # nie przerywamy pętli GUI z powodu wyjątku rysowania
            print(f"Błąd aktualizacji GUI: {e}")
        if not stop_flag:
            root.after(poll_ms, tk_update)

    # Start background preview thread (OpenCV windows)
    preview_t = threading.Thread(target=preview_loop, args=(root,), daemon=True)
    preview_t.start()

    # Start updating and enter mainloop
    root.after(100, tk_update)
    root.mainloop()


if __name__ == '__main__':
    # Jeżeli skrypt uruchomiony bezpośrednio, po wyjściu z pierwszej pętli
    # (po naciśnięciu 'q') wejdź tutaj i uruchom tkinter view.
    start_tkinter_board_view()