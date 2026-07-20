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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # Tách nền
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tiền"}
    
    # Lấy tờ tiền lớn nhất
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # 1. Tính toán tỷ lệ diện tích dựa trên Contour Area (Chính xác nhất cho tiền rách)
    dien_tich_thuc = cv2.contourArea(cnt)
    dien_tich_khung = w * h
    ty_le_contour = dien_tich_thuc / dien_tich_khung
    
    # 2. Chia lưới 50x50 để kiểm tra độ "đầy đủ" (chống việc tiền giả/vật thể lạ)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    crop = mask[y:y+h, x:x+w]
    
    rows, cols = 50, 50
    h_step, w_step = crop.shape[0] // rows, crop.shape[1] // cols
    so_o_con_nguyen = 0
    
    for r in range(rows):
        for c in range(cols):
            cell = crop[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
            if cv2.countNonZero(cell) > (h_step * w_step * 0.3): 
                so_o_con_nguyen += 1
    
    ty_le_luoi = so_o_con_nguyen / 2500
    
    # Kết hợp: Lấy giá trị thấp hơn để đảm bảo tính khắt khe
    ty_le = min(ty_le_contour, ty_le_luoi)
    
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le >= 0.60 else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le * 100, 2)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
