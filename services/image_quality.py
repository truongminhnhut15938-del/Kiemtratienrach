import cv2


def check_resolution(image):
    """
    Kiểm tra độ phân giải của ảnh.
    """

    height, width = image.shape[:2]

    shortest_side = min(width, height)

    if shortest_side >= 2000:
        level = "excellent"
        score = 100
        message = "Độ phân giải rất tốt."

    elif shortest_side >= 1200:
        level = "good"
        score = 80
        message = "Độ phân giải tốt."

    elif shortest_side >= 800:
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
    Kiểm tra độ mờ của ảnh bằng Variance of Laplacian.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if blur_score >= 300:
        level = "excellent"
        score = 100
        message = "Ảnh rất rõ nét."

    elif blur_score >= 150:
        level = "good"
        score = 80
        message = "Ảnh rõ."

    elif blur_score >= 80:
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
            "laplacian_variance": round(blur_score, 2)
        }
    }
