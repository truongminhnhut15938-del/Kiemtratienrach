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
