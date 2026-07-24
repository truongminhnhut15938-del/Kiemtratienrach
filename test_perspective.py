import cv2

from perspective_detection import detect_banknote


IMAGE_PATH = "test_images/test01.jpg"


def main():

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print(f"Không thể đọc ảnh: {IMAGE_PATH}")
        return

    result = detect_banknote(image)

    cv2.imshow("Original", image)

    if result is not None:

        cv2.imshow("Warped", result)

    else:

        print("Không phát hiện được tờ tiền.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()
