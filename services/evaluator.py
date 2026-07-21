EXCHANGE_THRESHOLD = 60.0


def evaluate_exchange_condition(ratio):
    """
    Đánh giá điều kiện thu đổi theo tỷ lệ diện tích còn lại.
    """

    eligible = ratio >= EXCHANGE_THRESHOLD

    return {
        "ratio": ratio,
        "threshold": EXCHANGE_THRESHOLD,
        "eligible": eligible,
        "message": (
            "Đủ điều kiện thu đổi"
            if eligible
            else "Không đủ điều kiện thu đổi"
        )
    }
