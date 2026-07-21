from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

# Kích thước chuẩn (mm) theo Ngân hàng Nhà nước để quy ra diện tích tỷ lệ chuẩn
CAU_HINH_CHUAN = {
    "500000": {"dai": 152, "rong": 65},
    "200000": {"dai": 148, "rong": 65},
    "100000": {"dai": 144, "rong": 65},
    "50000":  {"dai": 140, "rong": 65},
    "20000":  {"dai": 136, "rong": 65},
    "10000":  {"dai": 132, "rong": 60}
}

@app.post("/quet-tien")
async def quet_tien_api(file: UploadFile = File(...), menh_gia: str = Form(...)):
    if menh_gia not in CAU_HINH_CHUAN:
        return {"status": "error", "message": "Mệnh giá không hợp lệ!"}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "error", "message": "Ảnh không hợp lệ!"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Tiền xử lý ngưỡng Otsu để tách vật thể ra khỏi nền trắng
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tiền!"}
    
    cnt = max(contours, key=cv2.contourArea)
    
    # 1. Diện tích thực tế của mảnh tiền (tính bằng số pixel trên ảnh)
    dien_tich_thuc_pixel = cv2.contourArea(cnt)
    
    # 2. Lấy bounding box thực tế của mảnh tiền
    x, y, w, h = cv2.boundingRect(cnt)
    dien_tich_khung_bao = w * h
    
    # 3. Sử dụng tỷ lệ khung hình chuẩn của mệnh giá (Dài / Rộng)
    cfg = CAU_HINH_CHUAN[menh_gia]
    ty_le_chuan_w_h = cfg["dai"] / cfg["rong"] # Ví dụ 132/60 = 2.2 cho 10k
    
    # Dựa vào chiều cao (h) hiện tại của mảnh tiền, suy ra chiều dài lý tưởng nếu nó nguyên vẹn
    # Hoặc ngược lại, dùng chiều rộng (w) suy ra chiều cao lý tưởng
    # Để an toàn cho tờ tiền rách góc, ta lấy cạnh lớn hơn làm chuẩn:
    canh_lon = max(w, h)
    canh_nho = canh_lon / ty_le_chuan_w_h
    
    # Diện tích lý tưởng chuẩn (pixel) ứng với góc chụp thực tế hiện tại
    dien_tich_chuan_pixel = canh_lon * canh_nho
    
    # 4. Tính tỷ lệ phần trăm diện tích còn lại thực tế so với chuẩn
    ty_le_con_lai = (dien_tich_thuc_pixel / dien_tich_chuan_pixel) * 100
    ty_le_con_lai = min(ty_le_con_lai, 100.0) # Không vượt quá 100%
    
    # 5. Đưa ra kết luận (Theo quy định NHNN thường yêu cầu diện tích còn lại >= 60-90% tùy loại rách, ta để ngưỡng 80% an toàn)
    nguong_chuan = 60.0
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le_con_lai >= nguong_chuan else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"

    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le_con_lai, 2),
        "debug_info": {
            "dien_tich_thuc": dien_tich_thuc_pixel,
            "dien_tich_chuan_ly_tuong": round(dien_tich_chuan_pixel, 2)
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
