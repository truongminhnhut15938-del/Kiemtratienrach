"""
=========================================
Project : KiemTraTienRach

Module  : Image Quality Assessment

Stage   : G1
Part    : P1

Functions:
- check_resolution()
- check_blur()

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
