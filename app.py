from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

# Cấu hình phom chuẩn của các mệnh giá
CAU_HINH_GRID = {
    "500000": {"dai": 152, "rong": 65},
    "200000": {"dai": 148, "rong": 65},
    "100000": {"dai": 144, "rong": 65},
    "50000":  {"dai": 140, "rong": 65},
    "20000":  {"dai": 136, "rong": 65},
    "10000":  {"dai": 132, "rong": 60}
}

@app.post("/quet-tien")
async def quet_tien_api(file: UploadFile = File(...), menh_gia: str = Form(...)):
    if menh_gia not in CAU_HINH_GRID:
        return {"status": "error", "message": "Mệnh giá không hợp lệ!"}

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"status": "error", "message": "Ảnh không hợp lệ!"}

    # 1. Đọc ảnh và tiền xử lý bằng Otsu để tách phom toàn bộ vật thể
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Tìm viền lớn nhất (khung bao ngoài của cả tờ tiền bao gồm cả góc rách)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"status": "error", "message": "Không tìm thấy tờ tiền trong ảnh!"}
    
    c_lon_nhat = max(contours, key=cv2.contourArea)

    # 2. XÁC ĐỊNH KHUNG HỘP CHUẨN HOÀN HẢO (Bounding Box & Perspective Transform)
    rect = cv2.minAreaRect(c_lon_nhat)
    box = cv2.boxPoints(rect)
    box = np.int64(box)

    width = int(rect[1][0])
    height = int(rect[1][1])

    # Đảm bảo chiều rộng luôn là cạnh dài của tờ tiền
    if width < height:
        width, height = height, width

    pts_dst = np.array([[0, height-1], [0, 0], [width-1, 0], [width-1, height-1]], dtype="float32")

    pts_src = np.zeros((4, 2), dtype="float32")
    s = box.sum(axis=1)
    pts_src[1] = box[np.argmin(s)]
    pts_src[3] = box[np.argmax(s)]
    diff = np.diff(box, axis=1)
    pts_src[2] = box[np.argmin(diff)]
    pts_src[0] = box[np.argmax(diff)]

    # Nắn thẳng tờ tiền về khung phẳng hoàn hảo
    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    img_warped = cv2.warpPerspective(binary, M, (width, height))

    # 3. CHIA LƯỚI 50x50 VÀ ĐẾM ĐIỂM SỐ CHÍNH XÁC
    so_hang, so_cot = 50, 50
    h_o_vuong = height / so_hang
    w_o_vuong = width / so_cot

    so_o_hop_le = 0
    tong_so_o = so_hang * so_cot

    for r in range(so_hang):
        for c in range(so_cot):
            y1 = int(r * h_o_vuong)
            y2 = int((r + 1) * h_o_vuong)
            x1 = int(c * w_o_vuong)
            x2 = int((c + 1) * w_o_vuong)

            o_vuong = img_warped[y1:y2, x1:x2]
            
            if o_vuong.shape[0] > 0 and o_vuong.shape[1] > 0:
                pixel_trang = cv2.countNonZero(o_vuong)
                dien_tich_o = o_vuong.shape[0] * o_vuong.shape[1]

                # Ngưỡng 50%: Ô nào chứa trên 50% diện tích tờ tiền mới được tính hợp lệ
                if dien_tich_o > 0 and (pixel_trang / dien_tich_o) > 0.50:
                    so_o_hop_le += 1

    # 4. TÍNH TOÁN KẾT QUẢ KHOA HỌC
    ty_le_dien_tich = (so_o_hop_le / tong_so_o) * 100
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le_dien_tich >= 60.0 else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"

    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le_dien_tich, 2),
        "debug_so_o_hop_le": f"{so_o_hop_le}/{tong_so_o}"
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
