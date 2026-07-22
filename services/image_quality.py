"""
=========================================
Project : KiemTraTienRach

Module  : Image Quality Assessment

Stage   : G1
Part    : P1

Version : 1.0

Description:
Đánh giá chất lượng ảnh đầu vào trước
khi thực hiện các bước xử lý.

Functions:
- check_resolution()
- check_blur()
- check_brightness()
- check_contrast()
=========================================
"""

import cv2


# ==========================
# Resolution Threshold
# ==========================

RESOLUTION_EXCELLENT = 2000
RESOLUTION_GOOD = 1200
RESOLUTION_FAIR = 800


# ==========================
# Blur Threshold
# ==========================

BLUR_EXCELLENT = 300
BLUR_GOOD = 150
BLUR_FAIR = 80


# ==========================
# Brightness Threshold
# ==========================

BRIGHTNESS_EXCELLENT_MIN = 90
BRIGHTNESS_EXCELLENT_MAX = 180

BRIGHTNESS_GOOD_MIN = 70
BRIGHTNESS_GOOD_MAX = 210

BRIGHTNESS_FAIR_MIN = 50
BRIGHTNESS_FAIR_MAX = 230


# ==========================
# Contrast Threshold
# ==========================

CONTRAST_EXCELLENT = 70
CONTRAST_GOOD = 50
CONTRAST_FAIR = 30
def check_resolution(image):
    """
    Kiểm tra độ phân giải của ảnh.
    """

    height, width = image.shape[:2]

    shortest_side = min(width, height)

    if shortest_side >= RESOLUTION_EXCELLENT:

        level = "excellent"
        score = 100
        message = "Độ phân giải rất tốt."

    elif shortest_side >= RESOLUTION_GOOD:

        level = "good"
        score = 80
        message = "Độ phân giải tốt."

    elif shortest_side >= RESOLUTION_FAIR:

        level = "fair"
        score = 60
        message = "Độ phân giải trung bình."

    else:

        level = "poor"
        score = 30
        message = "Độ phân giải thấp."

    return {

        "score": score,

        "level": level,

        "message": message,

        "details": {

            "width": width,

            "height": height,

            "shortest_side": shortest_side

        }

    }


def check_blur(image):
    """
    Kiểm tra độ mờ của ảnh bằng
    Variance of Laplacian.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if blur_score >= BLUR_EXCELLENT:

        level = "excellent"
        score = 100
        message = "Ảnh rất rõ nét."

    elif blur_score >= BLUR_GOOD:

        level = "good"
        score = 80
        message = "Ảnh rõ."

    elif blur_score >= BLUR_FAIR:

        level = "fair"
        score = 60
        message = "Ảnh hơi mờ."

    else:

        level = "poor"
        score = 30
        message = "Ảnh bị mờ."

    return {

        "score": score,

        "level": level,

        "message": message,

        "details": {

            "laplacian_variance": round(
                blur_score,
                2
            )

        }

    }


def check_brightness(image):
    """
    Kiểm tra độ sáng trung bình của ảnh.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    brightness_score = gray.mean()

    if (
        BRIGHTNESS_EXCELLENT_MIN
        <= brightness_score <=
        BRIGHTNESS_EXCELLENT_MAX
    ):

        level = "excellent"
        score = 100
        message = "Độ sáng rất tốt."

    elif (
        BRIGHTNESS_GOOD_MIN
        <= brightness_score <
        BRIGHTNESS_EXCELLENT_MIN
    ) or (
        BRIGHTNESS_EXCELLENT_MAX <
        brightness_score <=
        BRIGHTNESS_GOOD_MAX
    ):

        level = "good"
        score = 80
        message = "Độ sáng tốt."

    elif (
        BRIGHTNESS_FAIR_MIN
        <= brightness_score <
        BRIGHTNESS_GOOD_MIN
    ) or (
        BRIGHTNESS_GOOD_MAX <
        brightness_score <=
        BRIGHTNESS_FAIR_MAX
    ):

        level = "fair"
        score = 60
        message = "Độ sáng chưa tối ưu."

    else:

        level = "poor"
        score = 30
        message = "Ảnh quá tối hoặc quá sáng."

    return {

        "score": score,

        "level": level,

        "message": message,

        "details": {

            "brightness": round(
                brightness_score,
                2
            )

        }

    }
def check_contrast(image):
    """
    Kiểm tra độ tương phản của ảnh.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    contrast_score = gray.std()

    if contrast_score >= CONTRAST_EXCELLENT:

        level = "excellent"
        score = 100
        message = "Độ tương phản rất tốt."

    elif contrast_score >= CONTRAST_GOOD:

        level = "good"
        score = 80
        message = "Độ tương phản tốt."

    elif contrast_score >= CONTRAST_FAIR:

        level = "fair"
        score = 60
        message = "Độ tương phản trung bình."

    else:

        level = "poor"
        score = 30
        message = "Độ tương phản thấp."

    return {

        "score": score,

        "level": level,

        "message": message,

        "details": {

            "contrast": round(
                contrast_score,
                2
            )

        }

    }
