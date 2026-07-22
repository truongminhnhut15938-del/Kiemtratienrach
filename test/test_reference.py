from services.reference import create_reference_mask

mask = create_reference_mask(500000)

if mask is None:
    print("Lỗi tạo Reference Mask")
else:
    print("Reference Mask tạo thành công")
