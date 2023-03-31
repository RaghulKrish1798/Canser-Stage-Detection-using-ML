### PURE FAILURE ###

import openslide
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import time

import cv2

dirpaths = glob('../dataset/*/*.svs')
print(dirpaths[0])

beginnning = time.time()
test_Slide = openslide.OpenSlide(dirpaths[9])
print(f"dimension: ({test_Slide.dimensions[0]/2}, {test_Slide.dimensions[1]/2})")

test_region = test_Slide.read_region((20000, 10000), 1, (300,300))
plt.imshow(test_region)
plt.show()

test_region = np.array(test_region)
test_region = test_region[:,:,:3]
# print(test_region)
white_value = np.sum(test_region, axis=2)
print(white_value)
white_num = np.sum(white_value >= 700)
print(white_num)
end_time = time.time()

print(f"time: {end_time - beginnning}")

# red_hist = cv2.calcHist(test_region, [0], None, [256], [0, 256])
# green_hist = cv2.calcHist(test_region, [1], None, [256], [0, 256])
# blue_hist = cv2.calcHist(test_region, [2], None, [256], [0, 256])
# white_hist = red_hist + green_hist + blue_hist
# print(white_hist.ravel())

# plt.hist(white_hist, color="blue")
# plt.show()