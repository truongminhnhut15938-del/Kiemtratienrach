from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

# Bảng kích thước chuẩn vật lý (mm) theo quy định của Ngân hàng Nhà nước
# Dùng để xác định tỷ lệ hình học chuẩn (Aspect Ratio) cho từng mệnh giá
THONG_SO_MENH_GIA = {
    "500000": {"dai": 152, "rong": 65, "aspect": 152 / 65},
    "200000": {"dai": 148, "rong": 65, "aspect": 148 / 65},
    "100000": {"dai": 144, "rong": 65, "aspect": 144 / 65},
    "50000":  {"dai": 140, "rong": 65, "aspect": 140 / 65},
    "20000":  {"dai": 136, "rong": 65, "aspect": 136 / 65},
    "10000":  {"dai": 132, "rong": 60, "aspect": 132 / 60}
}

@app.post("/quet-tien")
async def quet_tien_api(file: UploadFile = File(...), menh_gia: str = Form(...)):
    if menh_gia not in THONG_SO_MENH_GIA:
        return {"status": "error", "message": "Mệnh giá không hợp lệ!"}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "error", "message": "Ảnh không hợp lệ!"}

    # 1. Tiền xử lý ảnh chuyển xám và tách nền tối ưu bằng Otsu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Tìm tất cả các đường bao ngoài cùng của vật thể (tờ tiền)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tiền trong ảnh!"}
    
    # Lấy contour lớn nhất (mảnh tiền chính)
    c_lon_nhat = max(contours, key=cv2.contourArea)
    
    # Diện tích thực tế còn lại của mảnh tiền (tính bằng số pixel)
    dien_tich_thuc = cv2.contourArea(c_lon_nhat)

    # 3. Sử dụng minAreaRect để tìm hộp chữ nhật tối thiểu bao quanh mảnh tiền
    rect = cv2.minAreaRect(c_lon_nhat)
    box = cv2.boxPoints(rect)
    box = np.int64(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    if width == 0 or height == 0:
        return {"status": "error", "message": "Không thể xác định khung tiền!"}

    # Đảm bảo chiều rộng luôn là cạnh dài, chiều cao là cạnh ngắn
    if width < height:
        width, height = height, width

    # 4. TÍNH TOÁN DIỆN TÍCH CHUẨN ĐỘC LẬP THEO TỶ LỆ MỆNH GIÁ
    # Lấy thông số tỷ lệ chuẩn (Aspect Ratio) của mệnh giá được chọn
    aspect_chuan = THONG_SO_MENH_GIA[menh_gia]["aspect"]
    
    # Để tránh việc hộp bao bị bóp méo do rách góc, ta cố định cạnh dài hiện tại (width) 
    # và suy ra chiều cao lý tưởng thực tế dựa trên tỷ lệ chuẩn của mệnh giá đó.
    height_ly_tuong = width / aspect_chuan
    
    # Diện tích chuẩn lý tưởng của tờ tiền nguyên vẹn ở đúng góc chụp/khoảng cách này (pixel)
    dien_tich_chuan = width * height_ly_tuong

    # 5. Tính tỷ lệ phần trăm diện tích còn lại
    if dien_tich_chuan == 0:
        ty_le_con_lai = 0.0
    else:
        ty_le_con_lai = (dien_tich_thuc / dien_tich_chuan) * 100

    # Giới hạn trần tối đa là 100%
    ty_le_con_lai = min(ty_le_con_lai, 100.0)

    # 6. Thiết lập ngưỡng tiêu chuẩn thu đổi (Theo quy định chung thường từ 60% đến 80% diện tích)
    # Ta để ngưỡng an toàn là 60.0%
    NGUONG_THU_DOI = 60.0
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le_con_lai >= NGUONG_THU_DOI else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"

    return {
        "status": "success",
        "menh_gia": menh_gia,
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le_con_lai, 2),
        "debug_info": {
            "dien_tich_thuc": round(dien_tich_thuc, 2),
            "dien_tich_chuan": round(dien_tich_chuan, 2)
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
