from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

# Diện tích chuẩn (mm^2) theo tiêu chuẩn NHNN
# (Dài * Rộng)
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
    
    # 1. Tính diện tích thực (pixel)
    dien_tich_pixel = cv2.contourArea(cnt)
    
    # 2. Tính diện tích khung bao (pixel)
    dien_tich_khung_pixel = w * h
    
    # 3. Tỷ lệ diện tích tờ tiền so với khung bao (tiền polyme thường chiếm ~95-98% khung)
    # Tỷ lệ này giúp loại bỏ ảnh hưởng của khoảng cách chụp
    ty_le_chiem_dung = dien_tich_pixel / dien_tich_khung_pixel
    
    # Chuẩn hóa: Giả định tờ tiền nguyên vẹn chiếm 96% khung bao
    # Nếu tờ tiền rách, ty_le_chiem_dung sẽ giảm sâu
    ty_le_thuc = ty_le_chiem_dung / 0.96
    ty_le_thuc = min(ty_le_thuc, 1.0)
    
    # Kết luận
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le_thuc >= 0.70 else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le_thuc * 100, 2),
        "debug_info": {
            "area_pixel": dien_tich_pixel,
            "box_pixel": dien_tich_khung_pixel
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
