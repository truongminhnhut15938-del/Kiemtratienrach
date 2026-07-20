from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

# Tỷ lệ khung hình chuẩn của tiền polyme VN (Rộng/Cao) trung bình ~2.2
# Phương pháp: Dùng chiều rộng (w) làm mốc, diện tích chuẩn = w * (w / 2.2)
ASPECT_RATIO_CHUAN = 2.2 

@app.post("/quet-tien")
async def quet_tien_api(file: UploadFile = File(...), menh_gia: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "error", "message": "Ảnh không hợp lệ"}

    # Xử lý ảnh: làm nét và tách nền
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Tách tiền khỏi nền
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Tìm các đường bao (contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tiền"}
    
    # Lấy đường bao tờ tiền lớn nhất
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # --- LOGIC MỚI: TÍNH TỶ LỆ DỰA TRÊN DIỆN TÍCH CHUẨN ---
    # Thay vì lấy diện tích khung bao hiện tại (vốn đã bị rách), 
    # ta tính diện tích tờ tiền "lý tưởng" dựa trên chiều rộng (w)
    dien_tich_chuan = w * (w / ASPECT_RATIO_CHUAN)
    dien_tich_thuc = cv2.contourArea(cnt)
    
    # Tính tỷ lệ phần trăm còn lại
    ty_le = dien_tich_thuc / dien_tich_chuan
    
    # Giới hạn tối đa là 100% (tránh sai số nhiễu)
    ty_le = min(ty_le, 1.0)
    
    # Thiết lập ngưỡng: < 60% là không đủ điều kiện
    nguong_thu_doi = 0.60
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le >= nguong_thu_doi else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le * 100, 2)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
