from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np

app = FastAPI()

Cấu hình giá trị (Copy từ code cũ của bạn)
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
# Đọc dữ liệu ảnh từ request
contents = await file.read()
nparr = np.frombuffer(contents, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

if menh_gia not in CAU_HINH_GRID:
return {"error": "Menh gia khong hop le"}

# --- ĐOẠN LOGIC XỬ LÝ ẢNH CỦA BẠN (DÁN VÀO ĐÂY) ---
# Hãy chép các dòng xử lý từ hình 1000023495.jpg của bạn vào đây
# Ví dụ: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ...

# Kết quả trả về cho FlutterFlow:
return {
"ty_le_dien_tich": 75.5,
"ket_luan": "ĐỦ ĐIỀU KIỆN"
}