from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

# Tỷ lệ khung hình chuẩn của tiền polyme VN (Rộng/Cao) trung bình ~2.3
# Tờ 10k: 138x64 (~2.15), 100k: 148x65 (~2.27), 500k: 163x65 (~2.5)
ASPECT_RATIO_CHUAN = 2.3 

@app.post("/quet-tien")
async def quet_tien_api(file: UploadFile = File(...), menh_gia: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "error", "message": "Ảnh không hợp lệ"}

    # Xử lý ảnh để tách tiền
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Tìm đường bao
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tiền"}
    
    # Lấy contour lớn nhất (tờ tiền)
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # --- LOGIC MỚI: TÍNH TỶ LỆ DỰA TRÊN ASPECT RATIO ĐỂ CHỐNG GIAN LẬN RÁCH ---
    # Tính diện tích chuẩn dựa trên chiều rộng (w) vì nó ổn định hơn chiều cao (h) khi bị rách
    dien_tich_chuan = w * (w / ASPECT_RATIO_CHUAN)
    dien_tich_thuc = cv2.contourArea(cnt)
    
    # Tỷ lệ thực tế dựa trên hình học
    ty_le = dien_tich_thuc / dien_tich_chuan
    
    # Đảm bảo tỷ lệ không vượt quá 100% (do nhiễu hoặc sai số đo)
    ty_le = min(ty_le, 1.0)
    
    # Kết luận dựa trên ngưỡng 60%
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le >= 0.60 else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le * 100, 2)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
