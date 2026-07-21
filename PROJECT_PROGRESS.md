# PROJECT PROGRESS
# Dự án: KIEMTRATIENRACH

## Thông tin dự án

Tên dự án:
Kiemtratienrach

Ngôn ngữ:
Python 3

Framework:
Flask

Triển khai:
Render

Thư viện:

- OpenCV
- NumPy

--------------------------------------------

# Mục tiêu

Xây dựng hệ thống kiểm tra tiền Polymer Việt Nam bị rách.

Đầu vào:

Ảnh chụp tờ tiền.

Đầu ra:

- Tỷ lệ diện tích còn lại (%)
- Đủ điều kiện thu đổi
hoặc
- Không đủ điều kiện thu đổi

Điều kiện hiện tại:

Nếu tỷ lệ diện tích còn lại >= 60%

→ Đủ điều kiện thu đổi

Ngược lại

→ Không đủ điều kiện thu đổi

--------------------------------------------

# Phạm vi

Chỉ xử lý tiền Polymer Việt Nam.

Các mệnh giá:

- 500000
- 200000
- 100000
- 50000
- 20000
- 10000

--------------------------------------------

# Kiến trúc dự án

Kiemtratienrach/

app.py

config.py

services/

    perspective.py

    segmentation.py

    grid.py

    evaluator.py

templates/

static/

uploads/

tests/

PROJECT_PROGRESS.md

--------------------------------------------

# Luồng xử lý

Ảnh

↓

Perspective Transform

↓

Segmentation

↓

Grid 100 × 50

↓

Đếm số ô

↓

Tính tỷ lệ

↓

Đánh giá điều kiện thu đổi

↓

Hiển thị kết quả

--------------------------------------------

# TIẾN ĐỘ

## Module 1

Trạng thái:

✅ Hoàn thành

Nội dung:

- Khởi tạo Flask
- Upload ảnh
- Cấu trúc thư mục

--------------------------------------------

## Module 2

File:

services/perspective.py

Trạng thái:

✅ Hoàn thành

Đã xây dựng:

- resize_image()
- preprocess()
- enhance_edges()
- find_largest_contour()
- order_points()
- get_four_points()
- four_point_transform()
- detect_banknote()

--------------------------------------------

## Module 3

File:

services/segmentation.py

Trạng thái:

✅ Hoàn thành

Đã xây dựng:

- create_mask()
- keep_largest_component()
- apply_convex_hull()

Lưu ý:

Khi tính diện tích sẽ sử dụng
Mask thật.

Convex Hull chỉ hỗ trợ xử lý.

--------------------------------------------

## Module 4

Files:

services/grid.py

services/evaluator.py

Trạng thái:

✅ Hoàn thành

grid.py

- create_grid()
- draw_grid()
- count_remaining_cells()
- calculate_area_ratio()

evaluator.py

- evaluate_exchange_condition()

Điều kiện:

>=60%

Đủ điều kiện thu đổi.

--------------------------------------------

# MODULE HIỆN TẠI

➡ Module 5

Trạng thái:

🟡 Chưa bắt đầu

--------------------------------------------

# Module 5

Mục tiêu

Tích hợp toàn bộ các Module vào app.py.

Pipeline:

Upload

↓

Perspective

↓

Segmentation

↓

Grid

↓

Ratio

↓

Evaluator

↓

Hiển thị kết quả

--------------------------------------------

# Module 5 sẽ thực hiện

Part 1

Kết nối perspective.py

Part 2

Kết nối segmentation.py

Part 3

Kết nối grid.py

Part 4

Kết nối evaluator.py

Part 5

Hiển thị kết quả trên Web

Part 6

Kiểm thử toàn bộ hệ thống

--------------------------------------------

# Module 6 (Dự kiến)

- Nhận diện mệnh giá tự động

- Chuẩn hóa kích thước từng mệnh giá

- ORB Matching

- AI Segmentation

- REST API

- Docker

- Render Deployment

--------------------------------------------

# Quy ước phát triển

Sau mỗi Part

✔ Review code.

✔ Kiểm tra thuật toán.

✔ Chỉ chuyển sang Part tiếp theo khi đạt 100%.

--------------------------------------------

# Tiến độ

Module 1

██████████

100%

Module 2

██████████

100%

Module 3

██████████

100%

Module 4

██████████

100%

Module 5

░░░░░░░░░░

0%

--------------------------------------------

# BƯỚC TIẾP THEO

Module 5

Part 1

Tích hợp toàn bộ pipeline vào app.py.

--------------------------------------------

# Ghi chú

- Chỉ xử lý tiền Polymer.
- Chia lưới chuẩn 100 × 50.
- Tính tỷ lệ theo số ô còn lại.
- Ngưỡng thu đổi hiện tại: 60%.
- Mọi module đều được review trước khi chuyển sang module tiếp theo.
