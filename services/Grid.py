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
