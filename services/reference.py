import os
import cv2

from services.template_manager import (
    load_template,
    load_template_image
)
from services.normalizer import normalize_size
from services.segmentation import create_mask


def create_reference_mask(value):
    """
    Tạo Mask chuẩn từ ảnh mẫu.
    """

    metadata = load_template(value)

    if metadata is None:
        return None

    image = load_template_image(value)

    if image is None:
        return None

    normalized = normalize_size(
        image,
        metadata
    )

    mask = create_mask(normalized)

    output_path = os.path.join(
        "banknote_templates",
        str(value),
        "reference_mask.png"
    )

    cv2.imwrite(
        output_path,
        mask
    )

    return mask
