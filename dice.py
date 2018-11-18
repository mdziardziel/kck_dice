from cv2 import *
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np


nat = imread("dice01.jpg")

img = imread("dice01.jpg", 0)

kernel = np.ones((5,5),np.uint8)

alpha = 3
beta = 50
o1 = addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)

o2 = blur(o1,(5,5))

o3 = dilate(o2,kernel,iterations = 2)

o4 = erode(o3,kernel,iterations = 1)

_, o5 = threshold(o4,0,255,THRESH_BINARY + THRESH_OTSU)

_, cnt, _ = findContours(o5, RETR_TREE, CHAIN_APPROX_NONE)
 

drawContours(nat, cnt, -1, (0,255,0), 3)

numberofDice = 2
label = np.ones((len(cnt), 1), np.uint8)
label = np.ones((numberofDice, 1), np.uint8)

imshow("Orig", img)
imshow("Kontrast", o1)
imshow("Blur", o2)
imshow("Dilate", o3)
imshow("Erose", o4)
imshow("Otsu", o5)
imshow("Kontury", nat)

waitKey(0)
destroyAllWindows()
