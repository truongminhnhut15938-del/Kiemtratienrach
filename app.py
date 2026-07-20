from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

# BẢNG DIỆN TÍCH CHUẨN (tính bằng mm^2) dựa trên kích thước NHNN
# Diện tích = Dài * Rộng
BANG_DIEN_TICH_CHUAN_MM2 = {
    "10000": 132 * 60,
    "20000": 136 * 65,
    "50000": 140 * 65,
    "100000": 144 * 65,
    "200000": 148 * 65,
    "500000": 152 * 65
}

@app.post("/quet-tien")
async def quet_tien_api(file: UploadFile = File(...), menh_gia: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "error", "message": "Ảnh không hợp lệ"}

    # Xử lý ảnh
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tiền"}
    
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 1. TÍNH DIỆN TÍCH THỰC (pixel)
    dien_tich_pixel = cv2.contourArea(cnt)
    
    # 2. TÍNH TỶ LỆ DỰA TRÊN KHUNG BAO (w, h)
    # Vì ảnh chụp có khoảng cách khác nhau, ta dùng tỷ lệ khung hình chuẩn
    # để suy ra 1 pixel bằng bao nhiêu mm^2
    # Area_chuẩn_pixel = (w * h) * (tỷ lệ_đặc_điểm_tiền_so_với_khung)
    # Cách đơn giản: Tỷ lệ = Area_contour / Area_khung_bao * (Diện_tích_chuẩn_thực_tế_đã_biết)
    
    # Tuy nhiên, cách chuẩn xác nhất không dùng pixel là dùng tỷ lệ:
    # Tỷ lệ = Area_contour / Area_bounding_box
    # Và so sánh với một tờ tiền nguyên vẹn (thường chiếm ~0.95 - 0.98 diện tích khung bao)
    
    ty_le = dien_tich_pixel / (w * h)
    
    # Chuẩn hóa tỷ lệ: Tờ tiền nguyên vẹn ~ 0.95. 
    # Nếu rách, tỷ lệ này sẽ giảm xuống.
    ty_le_thuc = ty_le / 0.95 
    ty_le_thuc = min(ty_le_thuc, 1.0)
    
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le_thuc >= 0.60 else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le_thuc * 100, 2)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
