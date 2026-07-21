"""
Cấu hình hệ thống
"""

# ==========================
# Kích thước lưới
# ==========================

GRID_X = 100
GRID_Y = 50

# Một ô được xem là còn nếu diện tích còn >= 30%
CELL_THRESHOLD = 0.30

# Đủ điều kiện thu đổi nếu diện tích còn >= 60%
EXCHANGE_THRESHOLD = 0.60

# Thư mục lưu ảnh
UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

# Kích thước chuẩn tiền polymer Việt Nam (mm)
BANKNOTE_SIZE = {
    "500000": (152, 65),
    "200000": (148, 65),
    "100000": (144, 65),
    "50000":  (140, 65),
    "20000":  (136, 65),
    "10000":  (132, 60)
}

# Quy đổi mm → pixel
PIXELS_PER_MM = 5
