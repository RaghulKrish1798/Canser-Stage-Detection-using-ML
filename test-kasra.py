### PURE FAILURE ###

import openslide
import os
from glob import glob
import matplotlib.pyplot as plt

import cv2

dirpaths = glob('../dataset/*/*.svs')
print(dirpaths[0])

test_Slide = openslide.OpenSlide(dirpaths[9])
print(test_Slide.dimensions)

test_region = test_Slide.read_region((30000,10000), 0, (300,300))

blue_hist = cv2.calcHist(test_region, [0], None, [100], [0, 256])
plt.hist(blue_hist, color="blue")


print ("hey!")