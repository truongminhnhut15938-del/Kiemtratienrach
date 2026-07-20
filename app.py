from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

# BẢNG DIỆN TÍCH TIÊU CHUẨN (Quy đổi ra số ô trong lưới 50x50)
# Đây là "thước đo" cứng để đối soát mệnh giá
BANG_DIEN_TICH_CHUAN = {
    "10000": 2100,
    "20000": 2250,
    "50000": 2350,
    "100000": 2400,
    "200000": 2480,
    "500000": 2500
}

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
    
    # Tìm đường bao
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tiền"}
    
    # Lấy tờ tiền lớn nhất
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Tạo mask của tờ tiền
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    crop = mask[y:y+h, x:x+w]
    
    # Chia lưới 50x50 và đếm ô
    so_o_con_nguyen = 0
    rows, cols = 50, 50
    h_step, w_step = crop.shape[0] // rows, crop.shape[1] // cols
    
    for r in range(rows):
        for c in range(cols):
            cell = crop[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
            # Nếu ô có chứa phần tiền (trên 30% pixel là trắng)
            if cv2.countNonZero(cell) > (h_step * w_step * 0.3):
                so_o_con_nguyen += 1
                
    # Tính tỷ lệ dựa trên DIỆN TÍCH CHUẨN ĐÃ QUY ĐỊNH
    dien_tich_chuan = BANG_DIEN_TICH_CHUAN.get(menh_gia, 2400)
    ty_le = so_o_con_nguyen / dien_tich_chuan
    
    # Kết luận
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le >= 0.60 else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le * 100, 2),
        "debug_so_o": so_o_con_nguyen
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
