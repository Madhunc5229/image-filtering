import cv2
import numpy as np
from matplotlib import pyplot as plt
from common import *


image_path = 'data/GL_pyramid_data/butterfly.jpg'

image = read_image_as_gray(image_path)
cv2.imshow('original', toUint8(image))
cv2.waitKey(0)

cutoff_frequency = 1

G, L = pyramidsGL(image, 5, cutoff_frequency)

displayPyramids(G, L)

reconstructed = reconstructLaplacianPyramid(G, L, cutoff_frequency)
out = toUint8(reconstructed)
cv2.imwrite('output/GL_pyramid/reconstructedImage.png', out)
cv2.imshow('reconstructed', out)
cv2.waitKey(0)
