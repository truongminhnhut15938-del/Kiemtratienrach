# 1. Bảng diện tích tiêu chuẩn (được quy đổi ra số ô trong lưới 50x50)
# Giả sử tờ tiền nguyên vẹn luôn chiếm gần như toàn bộ 2500 ô
BANG_DIEN_TICH_CHUAN = {
    "10000": 2400,  # Ví dụ: tờ 10k chiếm 2400 ô
    "20000": 2400,
    "50000": 2400,
    "100000": 2450,
    "500000": 2480
}

@app.post("/quet-tien")
async def quet_tien_api(file: UploadFile = File(...), menh_gia: str = Form(...)):
    # ... (các bước xử lý ảnh và tạo mask giữ nguyên) ...
    
    # Lấy tờ tiền lớn nhất
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Chia lưới 50x50 trên vùng ảnh đã cắt
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    crop = mask[y:y+h, x:x+w]
    
    # Đếm số ô còn tiền
    so_o_con_nguyen = 0
    rows, cols = 50, 50
    h_step, w_step = crop.shape[0] // rows, crop.shape[1] // cols
    
    for r in range(rows):
        for c in range(cols):
            cell = crop[r*h_step:(r+1)*h_step, c*w_step:(c+1)*w_step]
            # Nếu ô có chứa phần tiền (trên 30% pixel là trắng)
            if cv2.countNonZero(cell) > (h_step * w_step * 0.3):
                so_o_con_nguyen += 1
                
    # 2. Tính tỷ lệ dựa trên DIỆN TÍCH CHUẨN ĐÃ QUY ĐỊNH
    dien_tich_chuan = BANG_DIEN_TICH_CHUAN.get(menh_gia, 2400)
    ty_le = so_o_con_nguyen / dien_tich_chuan
    
    ket_luan = "ĐỦ ĐIỀU KIỆN THU ĐỔI" if ty_le >= 0.60 else "KHÔNG ĐỦ ĐIỀU KIỆN THU ĐỔI"
    
    return {
        "status": "success",
        "ket_luan": ket_luan,
        "ty_le_con_lai": round(ty_le * 100, 2)
    }
