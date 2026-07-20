from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

@app.post("/quet-tien")
async def quet_tien_api(file: UploadFile = File(...), menh_gia: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "error", "message": "Ảnh không hợp lệ"}

    # Tiền polyme VN có đặc điểm tương phản cao với nền
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Dùng GaussianBlur để giảm nhiễu trước khi threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tiền"}
    
    # Lấy tờ tiền lớn nhất
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 1. Diện tích thực tế của mảnh tiền (pixel)
    dien_tich_thuc = cv2.contourArea(cnt)
    
    # 2. Diện tích khung bao (Bounding Box)
    dien_tich_bao = w * h
    
    # 3. Tính "Tỷ lệ lấp đầy" (Fill Ratio)
    # Tờ tiền nguyên vẹn lấp đầy khoảng 92-95% diện tích khung bao
    ty_le_lap_day = dien_tich_thuc / dien_tich_bao
    
    # 4. Chuẩn hóa về thang điểm 1.0 (Giả định tờ tiền chuẩn lấp đầy 0.92)
    ty_le_thuc = ty_le_lap_day / 0.92
    ty_le_thuc = min(ty_le_thuc, 1.0)
    
    # Ngưỡng 0.60 (60%) là đủ khắt khe cho tiền rách
    nguong_thu_doi = 0.60
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le_thuc >= nguong_thu_doi else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le_thuc * 100, 2),
        "debug_info": {
            "dien_tich_thuc": dien_tich_thuc,
            "dien_tich_bao": dien_tich_bao,
            "ty_le_lap_day": round(ty_le_lap_day, 4)
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
