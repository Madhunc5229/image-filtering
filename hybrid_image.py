# Import necessary packages here
import cv2
import numpy as np
from matplotlib import pyplot as plt
from common import *

image_path1 = 'data/hybrid_Images_data/makeup_before.jpg'

image_path2 = 'data/hybrid_Images_data/makeup_after.jpg'

image_1 = read_image_as_float(image_path1)
image_2 = read_image_as_float(image_path2)
cv2.imshow("Original image 1: ", cv2.imread(image_path1))
cv2.waitKey(0)
cv2.imshow("Original image 2: ", cv2.imread(image_path2))
cv2.waitKey(0)

row = np.ones((369, 1, 3))
col = np.ones((1, 250, 3))
image_1 = np.hstack((image_1, row))
image_1 = np.vstack((image_1, col))


cutoff_frequency = 2
filter_size = cutoff_frequency*4+1


filter = gaussian_2D_filter(filter_size, cutoff_frequency)

blurred_image1 = toUint8(imgfilter(image_1, filter=filter))

low_frequencies = blurred_image1

sharp_image_2 = image_2 - imgfilter(image_2, filter=filter)

high_frequencies = sharp_image_2

hybrid_image = low_frequencies + high_frequencies

img = toUint8(hybrid_image)
output = vis_hybrid_image(img)


cv2.imshow("Low frequency of image 1: ", blurred_image1)
cv2.waitKey(0)
cv2.imshow("High frequency of image 2: ", toUint8(sharp_image_2))
cv2.waitKey(0)
cv2.imshow('Hybrid image vizualization', output)
cv2.waitKey(0)


cv2.imwrite('output/hybrid_Images/low_frequency_image.png', blurred_image1)
cv2.imwrite('output/hybrid_Images/high_frequency_image.png',
            toUint8(sharp_image_2))
cv2.imwrite('output/hybrid_Images/hybrid_image.png', img)
cv2.imwrite('output/hybrid_Images/hybrid_image_vizualized.png', output)
