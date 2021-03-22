import numpy as np
import cv2 as cv
import pytesseract
import matplotlib.pyplot as plt

def print_hi(name):
    image = cv.imread("photo15.jpg")
    imageBGRGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    imageRGBGray = cv.cvtColor(imageBGRGray, cv.COLOR_BGR2RGB)
    shape = cv.imread("shape2.jpg")
    shapeBGRGray = cv.cvtColor(shape, cv.COLOR_BGR2GRAY)
    shapeRGBGray = cv.cvtColor(shapeBGRGray, cv.COLOR_BGR2RGB)
    normimageRGBGray = cv.normalize(imageRGBGray, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)#попробовать разделить на макс mageRGBGRAY
    normshapeRGBGray = cv.normalize(shapeRGBGray, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    h, w, c = normshapeRGBGray.shape[::]
    res = cv.matchTemplate(normimageRGBGray, normshapeRGBGray, cv.TM_CCOEFF_NORMED)
    threshold = 0.1
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(normimageRGBGray, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 1)
    image = cv.cvtColor(normimageRGBGray, cv.COLOR_BGR2RGB)
    plt.imshow(image)
    plt.show()

if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
