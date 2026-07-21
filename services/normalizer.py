import cv2


def normalize_size(image, metadata):
    """
    Chuẩn hóa ảnh theo kích thước quy định trong metadata.
    """

    width = metadata["resize_width"]
    height = metadata["resize_height"]

    normalized = cv2.resize(
        image,
        (width, height)
    )

    return normalized
