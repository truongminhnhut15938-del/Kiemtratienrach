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

    # 2. Xử lý ảnh: Chuyển sang xám và làm mờ để giảm nhiễu
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Tách nền (Thresh)
    # Dùng THRESH_BINARY_INV + OTSU để tờ tiền (thường là vùng sáng) nổi bật trên nền
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4. Tìm đường bao (Contour) lớn nhất
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tờ tiền"}
    
    cnt = max(contours, key=cv2.contourArea)
    
    # 5. Tạo Mask (Mặt nạ) để chỉ đếm diện tích tờ tiền
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    
    # 6. Chia lưới 10x10 trên vùng bao (Bounding Box) của tờ tiền
    x, y, w, h = cv2.boundingRect(cnt)
    # Lấy vùng tờ tiền đã mask
    roi_masked = cv2.bitwise_and(mask, mask, mask=mask)
    crop = roi_masked[y:y+h, x:x+w]
    
    rows, cols = 10, 10
    h_step, w_step = crop.shape[0] // rows, crop.shape[1] // cols
    
    so_o_con_nguyen = 0
    # Đếm số ô có diện tích > 20% (ngưỡng để coi là còn tiền)
    nguong_o = h_step * w_step * 0.2
    
    for r in range(rows):
        for c in range(cols):
            cell = crop[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
            if cv2.countNonZero(cell) > nguong_o:
                so_o_con_nguyen += 1
                
    # 7. Tính kết quả
    ty_le = so_o_con_nguyen / 100
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le >= 0.6 else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le * 100, 2)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
