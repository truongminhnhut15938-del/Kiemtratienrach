from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

# Tỷ lệ khung hình chuẩn của tiền polyme VN (Rộng/Cao) ~2.2
ASPECT_RATIO_CHUAN = 2.2 

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
    
    # Lấy đường bao tờ tiền lớn nhất
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 1. Tính diện tích thực tế của mảnh tiền (contourArea)
    dien_tich_thuc = cv2.contourArea(cnt)
    
    # 2. Tính "Diện tích lý tưởng" của tờ tiền nguyên vẹn
    # Sử dụng chiều cao (h) hiện tại của mảnh tiền làm thước đo 
    # Nếu tờ tiền còn nguyên, chiều dài phải là (h * ASPECT_RATIO_CHUAN)
    dien_tich_chuan = h * (h * ASPECT_RATIO_CHUAN)
    
    # 3. Tính tỷ lệ thực tế
    ty_le_thuc = dien_tich_thuc / dien_tich_chuan
    
    # Giới hạn tối đa là 1.0 (100%)
    ty_le_thuc = min(ty_le_thuc, 1.0)
    
    # 4. Ngưỡng thu đổi khắt khe (0.60 = 60%)
    nguong_thu_doi = 0.60
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le_thuc >= nguong_thu_doi else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le_thuc * 100, 2),
        "debug_info": {
            "dien_tich_thuc": dien_tich_thuc,
            "dien_tich_chuan": dien_tich_chuan
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
