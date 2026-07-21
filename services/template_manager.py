import os
import json
import cv2


TEMPLATE_ROOT = "banknote_templates"


def load_template(value):
    """
    Đọc thông tin của một mẫu tiền.
    """

    folder = os.path.join(
        TEMPLATE_ROOT,
        str(value)
    )

    metadata_path = os.path.join(
        folder,
        "metadata.json"
    )

    if not os.path.exists(metadata_path):
        return None

    with open(
        metadata_path,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return data


def load_template_image(
    value,
    side="front"
):
    """
    Đọc ảnh mẫu.
    """

    metadata = load_template(value)

    if metadata is None:
        return None

    filename = metadata[f"{side}_image"]

    image_path = os.path.join(
        TEMPLATE_ROOT,
        str(value),
        filename
    )

    return cv2.imread(image_path)
