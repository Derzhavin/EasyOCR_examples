import easyocr
import cv2
from glob import glob

from utility import draw_easy_ocr_result

user_network_directory = './user_network_directory'

reader = easyocr.Reader(['en'], user_network_directory=user_network_directory,
                        model_storage_directory=user_network_directory,
                        recog_network='number_plate_eu')

images_paths = glob('./custom_examples/*.jpg')

for image_path in images_paths:
    print('-' * 50)
    img = cv2.imread(image_path)
    img_h, img_w = img.shape[:2]
    scale_factor = 4
    img = cv2.resize(img, (img_w * scale_factor, img_h * scale_factor))

    result = reader.readtext(img)

    for text_area in result:
        print(text_area)

    draw_easy_ocr_result(img, result)

    cv2.imshow('result', img)
    cv2.waitKey(0)
