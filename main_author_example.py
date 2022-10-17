import easyocr
import cv2
from utility import draw_easy_ocr_result

reader = easyocr.Reader(['ch_sim','en'])
img = cv2.imread('./author_examples/chinese.jpg')

# reader = easyocr.Reader(['ko','en'])
# img = cv2.imread('./author_examples/korean.png')

# reader = easyocr.Reader(['fr'])
# img = cv2.imread('./author_examples/french.jpg')

# reader = easyocr.Reader(['en'])
# img = cv2.imread('./author_examples/english.png')

result = reader.readtext(img)

for text_area in result:
    print(text_area)

draw_easy_ocr_result(img, result)

cv2.imshow('result', img)
cv2.waitKey(0)

