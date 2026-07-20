from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

# Kích thước chuẩn (dùng để xác định tỷ lệ)
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

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. Tách tờ tiền ra khỏi nền (Dùng Contour để crop)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tờ tiền"}
    
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    roi = gray[y:y+h, x:x+w] # Cắt vùng tờ tiền
    
    # 2. Chia lưới 10x10 (Tổng 100 ô) trên vùng đã cắt
    rows, cols = 10, 10
    h_step, w_step = roi.shape[0] // rows, roi.shape[1] // cols
    
    so_o_con_nguyen = 0
    
    for r in range(rows):
        for c in range(cols):
            cell = roi[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
            # Nếu phần lớn ô là màu trắng (tờ tiền) thì coi là còn nguyên
            if cv2.countNonZero(cell) > (h_step * w_step * 0.5):
                so_o_con_nguyen += 1
                
    # 3. Kết luận dựa trên 60%
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
