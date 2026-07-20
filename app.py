@app.post("/quet-tien")
async def quet_tien_api(file: UploadFile = File(...), menh_gia: str = Form(...)):
    if menh_gia not in CAU_HINH_GRID:
        return {"status": "error", "message": "Mệnh giá không hợp lệ"}

    # Đọc ảnh từ file upload
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # 1. Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Làm mờ để giảm nhiễu (giúp Canny bắt cạnh chính xác hơn)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Dò cạnh bằng Canny
    edged = cv2.Canny(blurred, 50, 150)
    
    # 4. Tìm các đường bao (contours)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Nếu tìm thấy đường bao, lấy đường bao lớn nhất (tờ tiền)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        # Tính diện tích tờ tiền trên ảnh
        dien_tich_tien = cv2.contourArea(cnt)
        # Tính tổng diện tích bức ảnh
        total_area = img.shape[0] * img.shape[1]
        
        # Tính tỷ lệ diện tích tờ tiền so với toàn ảnh
        # Lưu ý: Tỷ lệ này sẽ rất nhỏ (khoảng 0.1 - 0.3) tùy vào góc chụp
        ty_le = dien_tich_tien / total_area
        
        # Ngưỡng (Threshold): 
        # Cần test thực tế xem tỷ lệ này là bao nhiêu thì đạt
        # Giả sử tờ tiền chiếm ít nhất 10% diện tích ảnh là hợp lệ
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
