from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

# 1. Khởi tạo app FastAPI đúng cách
app = FastAPI()

# Cấu hình mệnh giá (giữ nguyên cấu trúc của bạn)
CAU_HINH_GRID = {
    "500000": {'dai': 152, 'rong': 65},
    "200000": {'dai': 148, 'rong': 65},
    "100000": {'dai': 144, 'rong': 65},
    "50000": {'dai': 140, 'rong': 65},
    "20000": {'dai': 136, 'rong': 65},
    "10000": {'dai': 132, 'rong': 65}
}

@app.post("/quet-tien")
async def quet_tien_api(file: UploadFile = File(...), menh_gia: str = Form(...)):
    if menh_gia not in CAU_HINH_GRID:
        return {"status": "error", "message": "Mệnh giá không hợp lệ"}

    # Đọc dữ liệu ảnh từ file upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "error", "message": "Không thể đọc được file ảnh"}

    # 2. Xử lý ảnh bằng Canny Edge Detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    
    # 3. Tìm các đường bao (contours)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 4. Tính diện tích tờ tiền
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        dien_tich_tien = cv2.contourArea(cnt)
        total_area = img.shape[0] * img.shape[1]
        
        # Tỷ lệ diện tích tờ tiền/tổng ảnh
        ty_le = dien_tich_tien / total_area
        
        # Ngưỡng (Threshold): 0.05 là ví dụ, bạn có thể chỉnh lại sau khi test
        if ty_le >= 0.05: 
            ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI"
        else:
            ket_luan = "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
            
        return {
            "status": "success", 
            "ket_luan": ket_luan, 
            "ty_le_con_lai": round(ty_le * 100, 2)
        }
    else:
        return {"status": "error", "message": "Không tìm thấy tờ tiền trong ảnh"}

# 5. Chạy server với port động của Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
