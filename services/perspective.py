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

   
