import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

OPENSLIDE_PATH = r'C:/Users/RaghulKrish/Desktop/UB/Spring 23/CVIP/Project/openslide-win64-20221217/openslide-win64-20221217/bin'
if hasattr(os, 'add_dll_directory'):
    # Python >= 3.8 on Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

imgPath = 'C:/Users/RaghulKrish/Desktop/UB/Spring 23/CVIP/Project/Data/0e009ece-318f-4fcb-886e-cf3985c66e3b/TCGA-E9-A295-01A-01-TSA.14a39b7e-5d34-43ea-971c-09a8db7694e8.svs'
slide = openslide.OpenSlide(imgPath)
print(slide.dimensions)
print(slide.level_count)
print(slide.level_dimensions)

region = slide.read_region((1220, 2000), 2, (512, 512))
region = np.array(region)
# cv2.imwrite('test.png', region)
# image = cv2.imread('test.png', 0)
histogram = cv2.calcHist([region], [0], None, [256], [0, 256])
plt.hist(region.ravel(), 256, [0, 256])
plt.show()