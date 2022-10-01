import cv2
import numpy as np
from matplotlib import pyplot as plt
from common import* 


img_path = "data/edge_detection_inputs/102061.jpg" ## add the path here
img = cv2.imread(img_path)
cv2.imshow('original',img)
cv2.waitKey(0)
sigma = 1

M,theta = gradientMagnitude(img,sigma)
cv2.imshow('magnitude',M)
cv2.waitKey(0)
cv2.imwrite('output/edges/mag1.png',M)

out = edgeGradient(M,theta,canny=True)
cv2.imshow('canny using M',out)
cv2.waitKey(0)
cv2.imwrite('output/edges/out1.png',out)

magnitude = orientedFilterMagnitude(img,sigma)
out2 = edgeOrientedFilters(magnitude)

cv2.imshow('magnitude using oriented',magnitude)
cv2.waitKey(0)
cv2.imwrite('output/edges/mag2.png',magnitude)

cv2.imshow('canny using oriented filters',out2)
cv2.waitKey(0)
cv2.imwrite('output/edges/out2.png',out2)