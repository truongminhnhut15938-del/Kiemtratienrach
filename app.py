from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

@app.post("/quet-tien")
async def quet_tien_api(file: UploadFile = File(...), menh_gia: str = Form(...)):
    # 1. Đọc ảnh từ file upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "error", "message": "Ảnh không hợp lệ"}

    # 2. Xử lý ảnh: Chuyển sang xám và tăng tương phản cục bộ
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # 3. Tách nền bằng Adaptive Thresholding (ổn định hơn với các nền màu khác nhau)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 4. Tìm đường bao (Contour) của tờ tiền
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tờ tiền"}
    
    cnt = max(contours, key=cv2.contourArea)
    
    # 5. Tạo Mask và áp dụng để loại bỏ nền
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    roi_masked = cv2.bitwise_and(gray, gray, mask=mask)
    
    # Cắt vùng tờ tiền
    x, y, w, h = cv2.boundingRect(cnt)
    crop = roi_masked[y:y+h, x:x+w]
    
    # 6. Chia lưới 50x50 và đếm diện tích
    rows, cols = 50, 50
    h_step, w_step = crop.shape[0] // rows, crop.shape[1] // cols
    
    so_o_con_nguyen = 0
    # Ngưỡng đếm cho mỗi ô
    nguong_o = h_step * w_step * 0.15
    
    for r in range(rows):
        for c in range(cols):
            cell = crop[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
            # Dùng giá trị trung bình để xác định ô đó có tiền hay không
            if np.mean(cell) > 50: 
                so_o_con_nguyen += 1
                
    # 7. Tính toán tỷ lệ dựa trên 2500 ô
    ty_le = so_o_con_nguyen / 2500
    
    # Chuẩn thu đổi 60%
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le >= 0.60 else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le * 100, 2)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
