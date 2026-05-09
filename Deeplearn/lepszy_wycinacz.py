import cv2 as cv
import os
import glob
import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cut a board image into 8x8 square crops.")
    parser.add_argument("--input", default="7.png", help="Input image path.")
    parser.add_argument("--output", default="pola", help="Output directory.")
    parser.add_argument("--board-size", type=int, default=512, help="Resize target for board image.")
    parser.add_argument("--num-squares", type=int, default=8, help="Squares per row and column.")
    parser.add_argument("--prefix", default="pola_", help="Output filename prefix.")
    parser.add_argument("--ext", default="png", choices=["png", "jpg", "jpeg"], help="Output extension.")
    return parser.parse_args()


def resolve_input_path(input_value: str) -> Path:
    input_path = Path(input_value)
    if input_path.is_absolute() and input_path.exists():
        return input_path

    candidates = [
        Path.cwd() / input_path,
        Path(__file__).resolve().parent / input_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked_paths = "\n - ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Input image not found: '{input_value}'. Checked:\n - {checked_paths}"
    )


args = parse_args()
input_path = resolve_input_path(args.input)
img = cv.imread(str(input_path))
if img is None:
    raise ValueError(f"Could not read image file: {input_path}")

if args.board_size % args.num_squares != 0:
    raise ValueError("board-size must be divisible by num-squares.")

if img.shape[0] != args.board_size or img.shape[1] != args.board_size:
    img = cv.resize(img, (args.board_size, args.board_size), interpolation=cv.INTER_AREA)

square_size = args.board_size // args.num_squares
output_dir = args.output
os.makedirs(output_dir, exist_ok=True)

ext = args.ext.lower().lstrip(".")
if ext == "jpeg":
    ext = "jpg"

existing_files = glob.glob(os.path.join(output_dir, f"*.{ext}"))
max_num = 0
for file_path in existing_files:
    base = os.path.basename(file_path)
    if "_" not in base:
        continue
    suffix = base.split("_")[-1].split(".")[0]
    if suffix.isdigit():
        max_num = max(max_num, int(suffix))

counter = max_num + 1

for row in range(args.num_squares):
    for col in range(args.num_squares):
        y1 = row * square_size
        y2 = (row + 1) * square_size
        x1 = col * square_size
        x2 = (col + 1) * square_size

        crop = img[y1:y2, x1:x2]
        filename = os.path.join(output_dir, f"{args.prefix}{counter:05d}.{ext}")
        cv.imwrite(filename, crop)
        counter += 1

saved = counter - max_num - 1
print(
    f"Saved {saved} files ({square_size}x{square_size}) to '{output_dir}' "
    f"from '{input_path}'."
)