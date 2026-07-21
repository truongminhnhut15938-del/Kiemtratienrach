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

    new_size = (width, int(h * ratio))

    return cv2.resize(image, new_size)


def preprocess(image):
    """
    Tiền xử lý ảnh.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edge = cv2.Canny(blur, 50, 150)

    return edge
def enhance_edges(edge):
    """
    Tăng cường đường biên để xử lý
    tiền bị rách hoặc mất góc.
    """

    kernel = np.ones((5, 5), np.uint8)

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

    contours, _ = cv2.findContours(
        edge,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return None

    largest = max(contours, key=cv2.contourArea)

    return largest
def order_points(pts):
    """
    Sắp xếp 4 điểm theo thứ tự:
    Top-Left, Top-Right,
    Bottom-Right, Bottom-Left
    """

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)

    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)

    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def get_four_points(contour):
    """
    Lấy 4 góc của contour.
    """

    peri = cv2.arcLength(contour, True)

    approx = cv2.approxPolyDP(
        contour,
        0.02 * peri,
        True
    )

    # Nếu đã tìm được đúng 4 điểm
    if len(approx) == 4:
        pts = approx.reshape(4, 2)
        return order_points(pts)

    # Nếu không đủ 4 điểm thì dùng hình chữ nhật bao ngoài
    rect = cv2.minAreaRect(contour)

    box = cv2.boxPoints(rect)

    box = np.array(box, dtype="float32")

    return order_points(box)
def four_point_transform(image, pts):
    """
    Biến đổi phối cảnh (Perspective Transform)
    """

    rect = order_points(pts)

    (tl, tr, br, bl) = rect

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)

    maxWidth = max(int(widthA), int(widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(
        image,
        matrix,
        (maxWidth, maxHeight)
    )

    return warped
 


def detect_banknote(image):
    """
    Phát hiện và hiệu chỉnh tờ tiền.
    """

    img = resize_image(image)

    edge = preprocess(img)
    edge = enhance_edges (edge)

    contour = find_largest_contour(edge)

    if contour is None:
        return None

    points = get_four_points(contour)

    warped = four_point_transform(
        img,
        points
    )

    return warped   
    
