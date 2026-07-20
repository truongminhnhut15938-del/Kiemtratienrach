from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn

app = FastAPI()

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

    # Đọc ảnh
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Chuyển sang ảnh xám để xử lý
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Áp dụng ngưỡng để tìm tờ tiền (cần điều chỉnh tùy vào nền ảnh)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Chia ảnh thành các ô 100x100
    h, w = thresh.shape
    so_o_chua_tien = 0
    total_o = 0
    
    for y in range(0, h, 100):
        for x in range(0, w, 100):
            o_anh = thresh[y:y+100, x:x+100]
            # Nếu số pixel trắng trong ô vượt quá ngưỡng (có tờ tiền)
            if cv2.countNonZero(o_anh) > (100 * 100 * 0.3): # Ngưỡng 30% nội dung
                so_o_chua_tien += 1
            total_o += 1

    # Tính diện tích thực tế (tỉ lệ tương đối)
    dien_tich_thuc_te = so_o_chua_tien / total_o if total_o > 0 else 0
    
    # Giả sử diện tích chuẩn là 1.0 (100%), nếu diện tích thực > 60% thì đạt
    if dien_tich_thuc_te >= 0.6:
        ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI"
    else:
        ket_luan = "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success", 
        "ket_luan": ket_luan, 
        "ty_le_con_lai": round(dien_tich_thuc_te * 100, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
