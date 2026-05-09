import cv2 as cv
import numpy as np
import chess
import time
import threading
from collections import Counter, deque
import os
import json
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import tkinter as tk
from tkinter import font
import plansza

# === PARAMETRY ===
CAMERA_INDEX = 1
OUTPUT_SIZE = (512, 512)
ANALYSIS_INTERVAL = 0.8  # sekundy między analizami (tryb plynny)
CLASS_NAMES = ["black", "empty", "white"]
REQUIRED_CONSECUTIVE = 2  # bazowa liczba potwierdzen; dalej adaptowana dynamicznie
STATE_BUFFER_SIZE = 5
MIN_STABLE_VOTES = 3
LEGAL_MOVE_MAX_DIST = 6
LEGAL_MOVE_MIN_MARGIN = 1
RECOVERY_TRIGGER_STREAK = 3
RECOVERY_MAX_DIST = 4
RECOVERY_MIN_MARGIN = 1

# === MODEL ===
model = tf.keras.models.load_model('models/model_szachowy.keras')


def get_model_input_size(keras_model, fallback=(96, 96)):
    """Return (width, height) expected by the model input tensor."""
    shape = keras_model.input_shape
    # Some models expose a list of input shapes.
    if isinstance(shape, list):
        shape = shape[0]

    if not shape or len(shape) < 3:
        return fallback

    h, w = shape[1], shape[2]
    if h is None or w is None:
        return fallback

    return (int(w), int(h))


MODEL_INPUT_SIZE = get_model_input_size(model)
print(f"Model input size: {MODEL_INPUT_SIZE}")

# === STAN GLOBALNY ===
latest_state = None
prev_state = None
board = chess.Board()
lock = threading.RLock()
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
runtime_status = {
    "mode": "IDLE",
    "source": "-",
    "candidate": "-",
    "candidate_count": 0,
    "required_count": REQUIRED_CONSECUTIVE,
    "stable_unstable": 0,
    "legal_dist": None,
    "legal_margin": None,
    "illegal_streak": 0,
    "last_action": "-",
}


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
    global warped, analysis_counter, prev_state
    points = []
    transform_ready = False
    M = None
    vertical_lines = []
    horizontal_lines = []
    mode = 'none'
    warped = None
    analysis_counter = 0
    prev_state = None

    # Zamknij stare okno siatki, aby nie zostawac z "zamrozonym" poprzednim podgladem.
    try:
        cv.destroyWindow("Zaznacz linie")
    except Exception:
        pass

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
            tile = cv.resize(tile, MODEL_INPUT_SIZE).astype(np.float32) / 255.0

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


def board_to_color_state(board_obj):
    """Convert python-chess board into {square: black/white/empty} mapping."""
    result = {}
    for sq in chess.SQUARES:
        sq_name = chess.square_name(sq)
        piece = board_obj.piece_at(sq)
        if piece is None:
            result[sq_name] = "empty"
        elif piece.color == chess.WHITE:
            result[sq_name] = "white"
        else:
            result[sq_name] = "black"
    return result


def state_distance(state_a, state_b):
    """Simple Hamming distance between two color-state dictionaries."""
    return sum(1 for sq in state_a if state_a.get(sq) != state_b.get(sq))


def get_stable_state_from_buffer(state_buffer):
    """Aggregate last N observed states via per-square majority voting."""
    if len(state_buffer) < STATE_BUFFER_SIZE:
        return None, None

    stable = {}
    unstable_squares = 0

    for sq in state_buffer[-1].keys():
        labels = [st[sq] for st in state_buffer]
        counts = Counter(labels)
        top_label, top_votes = counts.most_common(1)[0]
        stable[sq] = top_label
        if top_votes < MIN_STABLE_VOTES:
            unstable_squares += 1

    # If too many squares are noisy, skip this cycle.
    if unstable_squares > 8:
        return None, unstable_squares

    return stable, unstable_squares


def get_required_consecutive(proposal, unstable_squares, illegal_streak):
    """Adaptive confirmation threshold: lower latency when confidence is high."""
    req = REQUIRED_CONSECUTIVE

    if proposal["type"] == "delta":
        # Direct source->target delta is usually most reliable.
        fr = proposal.get("fr")
        to = proposal.get("to")
        if fr != '?' and to != '?':
            req = max(2, req)
        else:
            req = max(3, req)
    else:
        dist = proposal.get("dist", 99)
        margin = proposal.get("margin", 0)
        if dist <= 2 and margin >= 2:
            req = 2
        else:
            req = 3

    # Raise threshold when scene is noisy or recent illegal streak occurred.
    if unstable_squares is not None and unstable_squares >= 5:
        req += 1
    if illegal_streak > 0:
        req += 1

    return min(req, 5)


def update_runtime_status(**kwargs):
    """Thread-safe partial updates for live debug overlay."""
    global runtime_status
    with lock:
        runtime_status.update(kwargs)


def draw_runtime_overlay(img):
    """Render lightweight debug HUD for live move-engine decisions."""
    with lock:
        st = dict(runtime_status)

    lines = [
        f"mode: {st.get('mode', '-')}",
        f"source: {st.get('source', '-')}",
        f"candidate: {st.get('candidate', '-')}",
        f"confirm: {st.get('candidate_count', 0)}/{st.get('required_count', REQUIRED_CONSECUTIVE)}",
        f"unstable: {st.get('stable_unstable', 0)}",
        f"legal d/m: {st.get('legal_dist', '-')}/{st.get('legal_margin', '-')}",
        f"illegal streak: {st.get('illegal_streak', 0)}",
        f"last: {st.get('last_action', '-')}",
    ]

    out = img.copy()
    x, y = 10, 24
    row_h = 18
    panel_w = 330
    panel_h = 12 + row_h * len(lines)
    overlay = out.copy()
    cv.rectangle(overlay, (x - 6, y - 18), (x - 6 + panel_w, y - 18 + panel_h), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.5, out, 0.5, 0, out)

    for i, txt in enumerate(lines):
        cv.putText(out, txt, (x, y + i * row_h), cv.FONT_HERSHEY_SIMPLEX, 0.5, (50, 255, 50), 1, cv.LINE_AA)

    return out


def infer_best_legal_move_from_state(board_obj, observed_state):
    """Find legal move whose resulting board best matches observed state."""
    scored = []
    for mv in board_obj.legal_moves:
        b = board_obj.copy(stack=False)
        b.push(mv)
        expected = board_to_color_state(b)
        dist = state_distance(expected, observed_state)
        scored.append((dist, mv))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    best_dist, best_move = scored[0]
    second_dist = scored[1][0] if len(scored) > 1 else best_dist + 1
    margin = second_dist - best_dist

    if best_dist > LEGAL_MOVE_MAX_DIST or margin < LEGAL_MOVE_MIN_MARGIN:
        return None

    piece = board_obj.piece_at(best_move.from_square)
    color = "white" if piece and piece.color == chess.WHITE else "black"
    return {
        "move": best_move,
        "dist": best_dist,
        "margin": margin,
        "color": color,
    }


def apply_move_object(mv):
    """Apply a validated python-chess Move object to global board."""
    global board
    if mv not in board.legal_moves:
        return False, f"Nielegalny ruch (obiekt): {mv.uci()}"
    san = board.san(mv)
    board.push(mv)
    return True, f"{mv.uci()} {san}"


def recover_board_from_observation(observed_state):
    """Try to re-sync board to camera observation using depth 1-2 legal move search."""
    global board
    candidates = []

    # Depth 1 candidates.
    for mv1 in board.legal_moves:
        b1 = board.copy(stack=False)
        b1.push(mv1)
        d1 = state_distance(board_to_color_state(b1), observed_state)
        candidates.append((d1, (mv1,)))

    # Depth 2 candidates.
    for mv1 in board.legal_moves:
        b1 = board.copy(stack=False)
        b1.push(mv1)
        for mv2 in b1.legal_moves:
            b2 = b1.copy(stack=False)
            b2.push(mv2)
            d2 = state_distance(board_to_color_state(b2), observed_state)
            candidates.append((d2, (mv1, mv2)))

    if not candidates:
        return False, "Brak kandydatow do recovery."

    candidates.sort(key=lambda x: x[0])
    best_dist, best_seq = candidates[0]
    second_dist = candidates[1][0] if len(candidates) > 1 else best_dist + 1
    margin = second_dist - best_dist

    if best_dist > RECOVERY_MAX_DIST or margin < RECOVERY_MIN_MARGIN:
        return False, f"Recovery niejednoznaczne (dist={best_dist}, margin={margin})."

    # Apply sequence to real board only after passing thresholds.
    applied = []
    for mv in best_seq:
        if mv not in board.legal_moves:
            return False, "Recovery przerwane: kandydat przestal byc legalny."
        san = board.san(mv)
        board.push(mv)
        applied.append(f"{mv.uci()} {san}")

    return True, f"Recovery zastosowane (dist={best_dist}): " + " | ".join(applied)

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
    print(f"▶️ Analiza co {ANALYSIS_INTERVAL}s uruchomiona.")

    # Stan bazowy i licznik do potwierdzania ruchu
    prev_state = None
    candidate_move = None
    candidate_count = 0
    illegal_streak = 0
    state_buffer = deque(maxlen=STATE_BUFFER_SIZE)

    while not stop_flag:
        update_runtime_status(mode="WAIT_FRAME")
        time.sleep(ANALYSIS_INTERVAL)
        if not transform_ready or len(vertical_lines) != 9 or len(horizontal_lines) != 9:
            update_runtime_status(mode="WAIT_CALIB", last_action="Brak gotowej kalibracji")
            continue
        with lock:
            img_copy = None if warped is None else warped.copy()
        if img_copy is None:
            update_runtime_status(mode="WAIT_WARP", last_action="Brak obrazu po transformacji")
            continue

        update_runtime_status(mode="ANALYZE")
        observed_state = analyze_board_with_model(img_copy, vertical_lines, horizontal_lines)
        if not observed_state:
            continue
        state_buffer.append(observed_state)

        stable_state, unstable_squares = get_stable_state_from_buffer(state_buffer)
        update_runtime_status(stable_unstable=0 if unstable_squares is None else unstable_squares)
        if stable_state is None:
            update_runtime_status(mode="STABILIZING", last_action="Czekam na stabilny stan")
            continue

        # Jeżeli nie mamy jeszcze stanu odniesienia, ustaw go i czekaj na kolejną iterację
        if prev_state is None:
            prev_state = stable_state
            candidate_move = None
            candidate_count = 0
            update_runtime_status(mode="BASELINE", last_action="Ustawiono stan bazowy")
            continue

        move = detect_move(prev_state, stable_state)
        proposal_key = None
        proposal = None

        if move is not None:
            fr, to, color = move
            proposal_key = ("delta", fr, to)
            proposal = {
                "type": "delta",
                "fr": fr,
                "to": to,
                "color": color,
            }
        else:
            with lock:
                legal_guess = infer_best_legal_move_from_state(board, stable_state)
            if legal_guess is not None:
                mv = legal_guess["move"]
                proposal_key = ("legal", mv.uci())
                proposal = {
                    "type": "legal",
                    "move": mv,
                    "color": legal_guess["color"],
                    "dist": legal_guess["dist"],
                    "margin": legal_guess["margin"],
                }

        if proposal is None:
            # brak wykrytego ruchu -> zresetuj licznik
            candidate_move = None
            candidate_count = 0
            update_runtime_status(
                mode="NO_MOVE",
                source="-",
                candidate="-",
                candidate_count=0,
                legal_dist=None,
                legal_margin=None,
                last_action="Brak ruchu",
            )
            continue

        required_count = get_required_consecutive(proposal, unstable_squares, illegal_streak)
        candidate_label = proposal_key[1] if proposal["type"] == "legal" else f"{proposal['fr']}->{proposal['to']}"
        update_runtime_status(
            mode="CANDIDATE",
            source=proposal["type"].upper(),
            candidate=candidate_label,
            required_count=required_count,
            legal_dist=proposal.get("dist"),
            legal_margin=proposal.get("margin"),
        )

        # Jeżeli wykryto ruch, porównaj z poprzednim kandydatem
        if proposal_key == candidate_move:
            candidate_count += 1
        else:
            candidate_move = proposal_key
            candidate_count = 1
        update_runtime_status(candidate_count=candidate_count)

        # Potwierdź ruch dopiero po wymaganej liczbie kolejnych detekcji
        if candidate_count >= required_count:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            # Spróbuj zaktualizować obiekt chess.Board().
            with lock:
                if proposal["type"] == "delta":
                    applied, desc = try_apply_move(proposal["fr"], proposal["to"])
                else:
                    applied, desc = apply_move_object(proposal["move"])

                if applied:
                    src = proposal["type"].upper()
                    log_line = f"{timestamp}  APPLIED[{src}]  {desc}"
                    move_history.append(log_line)
                    save_move_to_file(log_line)
                    print(f"♟️ Pewny ruch i zastosowano: {desc}")
                    illegal_streak = 0
                    update_runtime_status(
                        mode="COMMIT",
                        last_action=f"APPLIED[{src}] {desc}",
                        illegal_streak=illegal_streak,
                    )
                else:
                    illegal_streak += 1
                    print(f"⚠️ Potwierdzono detekcje, ale nie zastosowano: {desc}")
                    update_runtime_status(
                        mode="REJECTED",
                        last_action=f"Rejected: {desc}",
                        illegal_streak=illegal_streak,
                    )

                    if illegal_streak >= RECOVERY_TRIGGER_STREAK:
                        rec_ok, rec_desc = recover_board_from_observation(stable_state)
                        if rec_ok:
                            log_line = f"{timestamp}  RECOVERY  {rec_desc}"
                            move_history.append(log_line)
                            save_move_to_file(log_line)
                            print(f"🛠️ {rec_desc}")
                            illegal_streak = 0
                            update_runtime_status(
                                mode="RECOVERY_OK",
                                last_action=rec_desc,
                                illegal_streak=illegal_streak,
                            )
                        else:
                            print(f"⚠️ {rec_desc}")
                            update_runtime_status(
                                mode="RECOVERY_FAIL",
                                last_action=rec_desc,
                                illegal_streak=illegal_streak,
                            )

            # zresetuj i ustaw nowy stan odniesienia
            candidate_move = None
            candidate_count = 0
            prev_state = stable_state
        
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
    elif key in (ord('r'), ord('R')):
        reset_calibration(remove_saved=True)
        print("Reset kalibracji. Wybierz 4 rogi od nowa i ponownie ustaw linie.")
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
            display = draw_runtime_overlay(display)
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