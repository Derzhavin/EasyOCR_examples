import cv2


def draw_easy_ocr_result(img, easy_ocr_result):
    for text_area in easy_ocr_result:
        detected_area, text, _ = text_area

        top_left = (int(detected_area[0][0]), int(detected_area[0][1]))
        bottom_right = (int(detected_area[2][0]), int(detected_area[2][1]))
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)