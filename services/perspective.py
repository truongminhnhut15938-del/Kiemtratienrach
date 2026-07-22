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


def find_largest_contour(edge):
    """
    Tìm contour có diện tích lớn nhất.
    (Sẽ được nâng cấp ở B1P2G1.3)
    """

    contours, _ = cv2.findContours(
        edge,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(
        contours,
        key=cv2.contourArea
    )

    return largest


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

    contour = find_largest_contour(edge)

    if contour is None:
        return None

    points = get_four_points(contour)

    warped = four_point_transform(
        img,
        points
    )
   if warped is None
    return None
