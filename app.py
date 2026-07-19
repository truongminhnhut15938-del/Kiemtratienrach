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
    # Đọc ảnh
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # [DÁN ĐOẠN LOGIC XỬ LÝ ẢNH CỦA BẠN VÀO ĐÂY]
    # Ví dụ: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)...
    
    return {"status": "success", "ket_luan": "Đã xử lý xong"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
