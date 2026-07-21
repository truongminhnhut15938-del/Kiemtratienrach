import cv2
import numpy as np


def create_mask(image):
    """
    Tạo mask của phần tờ tiền còn lại.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, mask = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    kernel = np.ones((3, 3), np.uint8)

    # Loại bỏ nhiễu nhỏ
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )
mask = keep_largest_component(mask)
    # Lấp các lỗ nhỏ
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask
def keep_largest_component(mask):
    """
    Giữ lại vùng liên thông lớn nhất.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return mask

    largest = max(contours, key=cv2.contourArea)

    result = np.zeros_like(mask)

    cv2.drawContours(
        result,
        [largest],
        -1,
        255,
        thickness=cv2.FILLED
    )

    return result
