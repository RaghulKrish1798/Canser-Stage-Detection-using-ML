'''
Taking the data and slicing it up into patches
'''


### PURE FAILURE ###

import openslide
import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import time

import cv2

IMAGE_LEVEL = 1
PATCH_RES = 224

dirpaths = glob('../dataset/*/*.svs')
# print(dirpaths[0])



# print(test_Slide.level_downsamples)

dirnum = 0

for dirpath in dirpaths:
    dirnum += 1

    test_Slide = openslide.OpenSlide(dirpath)

    x_dim = int(test_Slide.dimensions[0]/test_Slide.level_downsamples[IMAGE_LEVEL])
    y_dim = int(test_Slide.dimensions[1]/test_Slide.level_downsamples[IMAGE_LEVEL])
    print(f"dimension: ({x_dim}, {y_dim})")

    total_image_patches = []

    # Performance Stats
    patch_counter = 0
    total_patches = (x_dim // PATCH_RES) * (y_dim // PATCH_RES)
    mostly_white_patches = 0


    os.mkdir(f'../extracted_patches/{dirpath[48:64]}')
    origin_begin = time.time()
    not_selected_wr = []
    not_selected_list = []

    # total_test_region = test_Slide.read_region((0, 0), IMAGE_LEVEL, (x_dim,y_dim))
    # plt.imshow(total_test_region)
    # plt.show()



    for x in range(0, x_dim, PATCH_RES):
        for y in range(0, y_dim, PATCH_RES):

            beginnning = time.time()

            patch_counter += 1

            # Extracting patch in (x, y) coordinate for the upper left pixel with width and height PATCH_RES
            test_region = test_Slide.read_region((x, y), IMAGE_LEVEL, (PATCH_RES,PATCH_RES))

            ## Testing purposes
            # plt.imshow(test_region)
            # plt.show()

            test_region = np.array(test_region)

            # Removing opacity dimension
            test_region = test_region[:,:,:3]

            # Summing up the values for R, G, and B channels to find the white pixels 
            white_value = np.sum(test_region, axis=2)
            # Actually finding the number of white pixels
            white_num = np.sum(white_value >= 675)
            # print(white_num)

            # Seeing if more than 45 percent of the pixels in a patch are white
            white_ratio = 100 * white_num / float(PATCH_RES**2)
            is_white = white_ratio > 45.0

            # if(patch_counter in not_selected_list):
            #     not_selected_wr.append(white_ratio)
            
            # Appending the patch to the overall patches extracted from the image
            if not is_white:
                total_image_patches.append(test_region)
                cv2.imwrite(f'../extracted_patches/{dirpath[48:64]}/patch_{patch_counter}.png', test_region)

            end_time = time.time()
            # slide_status = "Rejected" if is_white else "Accepted"


            # print(f"Patch {patch_counter}/{total_patches}, time: {end_time - beginnning}, status: {slide_status}")

    final_end = time.time()
    print(f'dir {dirnum} processed')
    print(f"Total patches: {patch_counter}")
    print(f"Overall time: {final_end - origin_begin}")
    print(f"Number of patches accepted = {len(total_image_patches)}")