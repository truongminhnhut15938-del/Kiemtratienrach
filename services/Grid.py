import cv2
import numpy as np


GRID_COLS = 100
GRID_ROWS = 50


def create_grid(image):
    """
    Tạo lưới chuẩn 100x50.
    """

    h, w = image.shape[:2]

    cell_width = w / GRID_COLS
    cell_height = h / GRID_ROWS

    return {
        "rows": GRID_ROWS,
        "cols": GRID_COLS,
        "cell_width": cell_width,
        "cell_height": cell_height,
        "width": w,
        "height": h
    }
def draw_grid(image, grid):
    """
    Vẽ lưới lên ảnh.
    """

    output = image.copy()

    h = grid["height"]
    w = grid["width"]

    cw = grid["cell_width"]
    ch = grid["cell_height"]

    for i in range(1, grid["cols"]):
        x = int(i * cw)

        cv2.line(
            output,
            (x, 0),
            (x, h),
            (0, 255, 0),
            1
        )

    for j in range(1, grid["rows"]):
        y = int(j * ch)

        cv2.line(
            output,
            (0, y),
            (w, y),
            (0, 255, 0),
            1
        )

    return output
    def count_remaining_cells(mask, grid, threshold=0.3):
    """
    Đếm số ô còn lại của tờ tiền.
    threshold = tỷ lệ pixel trắng tối thiểu để ô được tính là còn tồn tại.
    """

    rows = grid["rows"]
    cols = grid["cols"]

    cw = grid["cell_width"]
    ch = grid["cell_height"]

    remaining = 0

    for r in range(rows):

        for c in range(cols):

            x1 = int(c * cw)
            y1 = int(r * ch)

            x2 = int((c + 1) * cw)
            y2 = int((r + 1) * ch)

            cell = mask[y1:y2, x1:x2]

            if cell.size == 0:
                continue

            white_pixels = cv2.countNonZero(cell)

            total_pixels = cell.shape[0] * cell.shape[1]

            ratio = white_pixels / total_pixels

            if ratio >= threshold:
                remaining += 1

    return remaining
