def check_resolution(image):
    """
    Kiểm tra độ phân giải của ảnh.
    """

    height, width = image.shape[:2]

    shortest_side = min(width, height)

    if shortest_side >= 2000:
        level = "excellent"

    elif shortest_side >= 1200:
        level = "good"

    elif shortest_side >= 800:
        level = "fair"

    else:
        level = "poor"

    return {
        "width": width,
        "height": height,
        "shortest_side": shortest_side,
        "level": level
    }
  def check_blur(image):
    """
    Kiểm tra độ mờ của ảnh bằng Variance of Laplacian.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    if score >= 300:
        level = "excellent"

    elif score >= 150:
        level = "good"

    elif score >= 80:
        level = "fair"

    else:
        level = "poor"

    return {
        "score": round(score, 2),
        "level": level
    }
