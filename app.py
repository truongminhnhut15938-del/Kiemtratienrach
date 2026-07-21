import os
import cv2

from flask import Flask, render_template, request

from services.perspective import detect_banknote
from services.segmentation import create_mask
from services.grid import (
    create_grid,
    count_remaining_cells,
    calculate_area_ratio
)
from services.evaluator import evaluate_exchange_condition


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():

    if "image" not in request.files:
        return "Không tìm thấy ảnh."

    file = request.files["image"]

    if file.filename == "":
        return "Chưa chọn ảnh."

    filepath = os.path.join(
        app.config["UPLOAD_FOLDER"],
        file.filename
    )

    file.save(filepath)

    image = cv2.imread(filepath)

    if image is None:
        return "Không đọc được ảnh."

    # Bước 1
    warped = detect_banknote(image)

    if warped is None:
        return "Không phát hiện được tờ tiền."

    # Bước 2
    mask = create_mask(warped)

    # Bước 3
    grid = create_grid(mask)

    remaining = count_remaining_cells(
        mask,
        grid
    )

    # Bước 4
    ratio = calculate_area_ratio(
        remaining
    )

    # Bước 5
    result = evaluate_exchange_condition(
        ratio
    )

    return render_template(
        "result.html",
        ratio=result["ratio"],
        eligible=result["eligible"],
        message=result["message"],
        remaining=remaining,
        total=5000
    )


if __name__ == "__main__":
    app.run(debug=True)
