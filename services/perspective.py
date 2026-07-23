"""
=========================================
Project : KiemTraTienRach
Module  : Perspective Detection
Stage   : G1
Part    : P2
Version : 1.1
=========================================
"""


import cv2
import numpy as np


def resize_image(image, width=1200):
    """
    Thu nhỏ ảnh để tăng tốc xử lý.
    """

    h, w = image.shape[:2]

    if w <= width:
        return image

    ratio = width / float(w)

    new_size = (
        width,
        int(h * ratio)
    )

    return cv2.resize(image, new_size)


def enhance_contrast(gray):
    """
    Tăng cường độ tương phản bằng CLAHE.

    CLAHE giúp làm nổi bật chi tiết
    và đường biên của tờ tiền trong
    nhiều điều kiện ánh sáng khác nhau.
    """

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    return enhanced


def preprocess(image):
    """
    Tiền xử lý ảnh.

    Pipeline:

    BGR
      ↓
    Gray
      ↓
    CLAHE
      ↓
    Gaussian Blur
      ↓
    Canny
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = enhance_contrast(gray)

    blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edge = cv2.Canny(
        blur,
        50,
        150
    )

    return edge


def enhance_edges(edge):
    """
    Tăng cường đường biên để xử lý
    tiền bị rách hoặc mất góc.
    """

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    edge = cv2.morphologyEx(
        edge,
        cv2.MORPH_CLOSE,
        kernel
    )

    edge = cv2.dilate(
        edge,
        kernel,
        iterations=1
    )

    return edge


# TODO:
# B1P2G1.3
# Thay thuật toán chọn contour lớn nhất
# bằng thuật toán chấm điểm contour.


# ==========================
# Contour Scoring Threshold
# ==========================

MIN_CONTOUR_AREA = 5000

AREA_EXCELLENT = 80000
AREA_GOOD = 50000
AREA_FAIR = 25000

RECTANGLE_EXCELLENT = 0.95
RECTANGLE_GOOD = 0.90
RECTANGLE_FAIR = 0.80

# ==========================
# Aspect Ratio Threshold
# ==========================

ASPECT_RATIO_GOOD_LOW = 2.15
ASPECT_RATIO_GOOD_HIGH = 2.45
ASPECT_RATIO_FAIR_LOW = 2.00
ASPECT_RATIO_FAIR_HIGH = 2.60

# ==========================
# Solidity Threshold
# ==========================

SOLIDITY_EXCELLENT = 0.98
SOLIDITY_GOOD = 0.95
SOLIDITY_FAIR = 0.90

# ==========================
# Border Distance Threshold
# ==========================

BORDER_MARGIN = 20

# ==========================
# Extent Threshold
# ==========================

EXTENT_EXCELLENT = 0.95
EXTENT_GOOD = 0.90
EXTENT_FAIR = 0.80


def score_rectangle(contour):
    """
    Chấm điểm mức độ giống hình chữ nhật
    của contour.
    """

    area = cv2.contourArea(contour)

    if area == 0:

        return {
            "score": 0,
            "rectangle_ratio": 0
        }

    rect = cv2.minAreaRect(contour)

    box = cv2.boxPoints(rect)

    box = np.array(
        box,
        dtype="float32"
    )

    rectangle_area = cv2.contourArea(box)

    if rectangle_area == 0:

        return {
            "score": 0,
            "rectangle_ratio": 0
        }

    ratio = area / rectangle_area

    if ratio >= RECTANGLE_EXCELLENT:

        score = 25

    elif ratio >= RECTANGLE_GOOD:

        score = 20

    elif ratio >= RECTANGLE_FAIR:

        score = 10

    else:

        score = 0

    return {

        "score": score,

        "rectangle_ratio": round(
            ratio,
            3
        )

    }


def score_aspect_ratio(contour):
    """
    Chấm điểm theo tỷ lệ dài/rộng
    của contour.
    """

    rect = cv2.minAreaRect(contour)

    width, height = rect[1]

    if width == 0 or height == 0:

        return {
            "score": 0,
            "aspect_ratio": 0
        }

    ratio = max(width, height) / min(width, height)

    if (
        ASPECT_RATIO_GOOD_LOW
        <= ratio <=
        ASPECT_RATIO_GOOD_HIGH
    ):

        score = 20

    elif (
        ASPECT_RATIO_FAIR_LOW
        <= ratio <=
        ASPECT_RATIO_FAIR_HIGH
    ):

        score = 10

    else:

        score = 0

    return {

        "score": score,

        "aspect_ratio": round(
            ratio,
            3
        )

    }


def score_solidity(contour):
    """
    Chấm điểm theo độ đặc (Solidity).

    Solidity =
    Contour Area / Convex Hull Area

    Giá trị càng gần 1 thì contour
    càng ít bị khuyết.
    """

    area = cv2.contourArea(contour)

    hull = cv2.convexHull(contour)

    hull_area = cv2.contourArea(hull)

    if hull_area == 0:

        return {
            "score": 0,
            "solidity": 0
        }

    solidity = area / hull_area

    if solidity >= SOLIDITY_EXCELLENT:

        score = 15

    elif solidity >= SOLIDITY_GOOD:

        score = 12

    elif solidity >= SOLIDITY_FAIR:

        score = 6

    else:

        score = 0

    return {

        "score": score,

        "solidity": round(solidity, 3)

    }


def score_extent(contour):
    """
    Chấm điểm theo Extent.

    Extent =
    Contour Area / Bounding Rect Area

    Giá trị càng gần 1 thì contour
    càng lấp đầy hình chữ nhật.
    """

    area = cv2.contourArea(contour)

    x, y, w, h = cv2.boundingRect(contour)

    rect_area = w * h

    if rect_area == 0:

        return {
            "score": 0,
            "extent": 0
        }

    extent = area / rect_area

    if extent >= EXTENT_EXCELLENT:

        score = 10

    elif extent >= EXTENT_GOOD:

        score = 8

    elif extent >= EXTENT_FAIR:

        score = 4

    else:

        score = 0

    return {

        "score": score,

        "extent": round(extent, 3)

    }
def score_border_distance(contour, image_shape):
    """
    Chấm điểm theo khoảng cách
    tới mép ảnh.

    Tờ tiền nằm trọn trong ảnh
    sẽ được điểm cao hơn.
    """

    h, w = image_shape[:2]

    x, y, cw, ch = cv2.boundingRect(contour)

    if (
        x <= BORDER_MARGIN or
        y <= BORDER_MARGIN or
        x + cw >= w - BORDER_MARGIN or
        y + ch >= h - BORDER_MARGIN
    ):

        score = 0

    else:

        score = 10

    return {

        "score": score,

        "bounding_box": (
            x,
            y,
            cw,
            ch
        )

    }


def score_area(contour):
    """
    Chấm điểm theo diện tích contour.
    """

    area = cv2.contourArea(contour)

    if area < MIN_CONTOUR_AREA:

        return {

            "score": 0,

            "area": round(area, 2)

        }

    if area >= AREA_EXCELLENT:

        score = 30

    elif area >= AREA_GOOD:

        score = 25

    elif area >= AREA_FAIR:

        score = 15

    else:

        score = 5

    return {

        "score": score,

        "area": round(area, 2)

    }


def find_best_contour(edge,image_shape):
    """
    Chọn contour có tổng điểm cao nhất.
    """

    contours, _ = cv2.findContours(
        edge,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    best_contour = None
    best_score = -1

    for index,contour in enumerate(contours, start=1):

        area_info = score_area(contour)

        if area_info["score"] == 0:
            continue

        rectangle_info = score_rectangle(contour)

        aspect_info = score_aspect_ratio(contour)

        solidity_info= score_solidity(contour)

        border_info = score_border_distance(
    contour,
    image_shape
        )

        extent_info = score_extent(contour)

        
total_score = (
            area_info["score"] +
            rectangle_info["score"]+
            aspect_info["score"]+
            solidity_info["score"]+
            border_info["score"]+
            extent_info["score"]
        )

        print(f"\n===== Contour #{index} =====")

        print(f"Area Score       : {area_info['score']}")
        print(f"Rectangle Score  : {rectangle_info['score']}")
        print(f"Aspect Score     : {aspect_info['score']}")
        print(f"Solidity Score   : {solidity_info['score']}")
        print(f"Border Score     : {border_info['score']}")
        print(f"Extent Score     : {extent_info['score']}")

        print("----------------------------")

        print(f"TOTAL SCORE      : {total_score}")

        if total_score > best_score:

            best_score = total_score

            best_contour = contour

            print(f"--> Contour #{index} đang dẫn đầu với {best_score} điểm")

    return best_contour

def order_points(pts):
    """
    Sắp xếp 4 điểm theo thứ tự:

    Top-Left
    Top-Right
    Bottom-Right
    Bottom-Left
    """

    rect = np.zeros(
        (4, 2),
        dtype="float32"
    )

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(
        pts,
        axis=1
    )

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_four_points(contour):
    """
    Lấy 4 góc của contour.
    """

    peri = cv2.arcLength(
        contour,
        True
    )

    approx = cv2.approxPolyDP(
        contour,
        0.02 * peri,
        True
    )

    if len(approx) == 4:

        pts = approx.reshape(
            4,
            2
        )

        return order_points(pts)

    rect = cv2.minAreaRect(contour)

    box = cv2.boxPoints(rect)

    box = np.array(
        box,
        dtype="float32"
    )

    return order_points(box)


def four_point_transform(image, pts):
    """
    Biến đổi phối cảnh.
    """

    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(
        br - bl
    )

    width_b = np.linalg.norm(
        tr - tl
    )

    max_width = max(
        int(width_a),
        int(width_b)
    )

    height_a = np.linalg.norm(
        tr - br
    )

    height_b = np.linalg.norm(
        tl - bl
    )

    max_height = max(
        int(height_a),
        int(height_b)
    )


    if max_width <= 0 or max_height <= 0:
        return None
        
    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ],
        dtype="float32"
    )

    matrix = cv2.getPerspectiveTransform(
        rect,
        dst
    )

    warped = cv2.warpPerspective(
        image,
        matrix,
        (
            max_width,
            max_height
        )
    )

    return warped


def detect_banknote(image):
    """
    Phát hiện và hiệu chỉnh
    phối cảnh của tờ tiền.
    """

    img = resize_image(image)

    edge = preprocess(img)

    edge = enhance_edges(edge)

    contour = find_best_contour(edge,img.shape)

    if contour is None:
        return None

    points = get_four_points(contour)

    warped = four_point_transform(
        img,
        points
    )
    if warped is None:
        return None
        
    return warped
