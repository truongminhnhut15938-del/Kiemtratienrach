from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

# Kích thước tiêu chuẩn vật lý chính xác của Ngân hàng Nhà nước
THONG_SO_MENH_GIA = {
    "500000": {"dai": 152.0, "rong": 65.0},
    "200000": {"dai": 148.0, "rong": 65.0},
    "100000": {"dai": 144.0, "rong": 65.0},
    "50000":  {"dai": 140.0, "rong": 65.0},
    "20000":  {"dai": 136.0, "rong": 65.0},
    "10000":  {"dai": 132.0, "rong": 60.0}
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

    # 1. Chuyển xám và tách nền bằng Otsu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # 2. Tìm contour lớn nhất của tờ tiền
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tiền trong ảnh!"}
    
    cnt = max(contours, key=cv2.contourArea)
    
    # 3. Sử dụng xấp xỉ đa giác (approxPolyDP) để tìm đúng 4 góc biên ngoài cùng của mảnh tiền
    # Dù rách chéo góc, đường viền ngoài vẫn tạo thành một đa giác khép kín chứa các đỉnh góc
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
    
    # Nếu tìm được 4 góc hoặc dùng minAreaRect làm khung gốc để lấy 4 điểm bao trùm tối đa
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    
    # Sắp xếp 4 góc chuẩn xác (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    pts_src = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    pts_src[1] = box[np.argmin(s)] # Top-Left
    pts_src[3] = box[np.argmax(s)] # Bottom-Right
    diff = np.diff(box, axis=1)
    pts_src[2] = box[np.argmin(diff)] # Top-Right
    pts_src[0] = box[np.argmax(diff)] # Bottom-Left

    # 4. Thiết lập kích thước khung chuẩn tuyệt đối dựa trên tỷ lệ thực tế của mệnh giá
    cfg = THONG_SO_MENH_GIA[menh_gia]
    ty_le_chuan = cfg["dai"] / cfg["rong"]
    
    # Cố định chiều rộng (width) theo hộp bao lớn nhất, suy ra chiều cao chuẩn lý tưởng
    width = int(max(np.linalg.norm(pts_src[1] - pts_src[2]), np.linalg.norm(pts_src[0] - pts_src[3])))
    height = int(width / ty_le_chuan)
    
    if width <= 0 or height <= 0:
        return {"status": "error", "message": "Lỗi kích thước khung tiền!"}

    # Tọa độ đích cho khung phẳng hoàn hảo tỷ lệ chuẩn
    pts_dst = np.array([
        [0, height - 1],
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1]
    ], dtype="float32")

    # Nắn phẳng tờ tiền về khung tỷ lệ chuẩn của NHNN
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    img_warped = cv2.warpPerspective(thresh, M, (width, height))

    # 5. TÍNH TOÁN DIỆN TÍCH THỰC TẾ TRÊN KHUNG CHUẨN ĐÃ NẮN PHẲNG
    # Toàn bộ khung chuẩn có diện tích pixel tối đa là width * height (tương ứng 100% tờ tiền nguyên vẹn)
    dien_tich_chuan_khung = width * height
    
    # Diện tích pixel thực tế còn lại của mảnh tiền bên trong khung chuẩn
    dien_tich_thuc_pixel = cv2.countNonZero(img_warped)

    if dien_tich_chuan_khung == 0:
        ty_le_con_lai = 0.0
    else:
        ty_le_con_lai = (dien_tich_thuc_pixel / dien_tich_chuan_khung) * 100

    ty_le_con_lai = min(ty_le_con_lai, 100.0)

    # 6. Ngưỡng xét duyệt thu đổi (Thường quy định từ 60% - 75% tùy ngân hàng)
    NGUONG_THU_DOI = 60.0
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le_con_lai >= NGUONG_THU_DOI else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"

    return {
        "status": "success",
        "menh_gia": menh_gia,
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le_con_lai, 2),
        "debug_info": {
            "dien_tich_thuc": int(dien_tich_thuc_pixel),
            "dien_tich_chuan": int(dien_tich_chuan_khung)
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
