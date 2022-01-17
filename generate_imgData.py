import random
import cv2 as cv
import matplotlib.pyplot as plt

# def show_in_plt(img):
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(img[:,:,[2, 1, 0]])
    
img = cv.imread('full_duck.jpg')
# show_in_plt(img)

height, width = img.shape[:2]
# print(height, width)

seed_v = random.random()
seed_h = random.random()

fetch_v_begin = int(seed_v * (height-500))
fetch_h_begin = int(seed_h * (width-500))

# print(fetch_v_begin, fetch_h_begin)

img_fetch = img[fetch_v_begin:fetch_v_begin+500, fetch_h_begin:fetch_h_begin+500]
# show_in_plt(img_fetch)

cv.imwrite('04duck.jpg', img_fetch)