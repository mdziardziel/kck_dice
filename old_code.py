from cv2 import *
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import os


on_paper = "IMG_20181118_220206.jpg"
on_table = "IMG_20181118_220153.jpg"


def show_image(title, image):
    namedWindow(title, WINDOW_NORMAL)
    imshow(title, image)



##########################3

    # show_image("o4", o4)
    # show_image("neg", neg)
    # cv2.drawContours(nat_resized, contours, -1, (0,255,0), 3)
    #
    # show_image('sob', u8lap)
    # show_image('trsh', thresh)
    # show_image('cont', nat_resized)
    # show_image('cont', img_resized)




        # gray = cv2.bilateralFilter(img_resized, 11, 17, 17)
        # laplacian = cv2.Sobel(gray,cv2.CV_64F,1,1,ksize=5)
        # abs_lap = np.absolute(laplacian)
        # # thresh = cv2.Canny(laplacian, 30, 200)
        # u8lap = np.uint8(abs_lap)
        # # u8lap[u8lap > 100] = 255
        # ret, thresh = cv2.threshold(u8lap,127,255,0)
        # im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)









#################################



# def trunc(value):
#     if value > 255: return 255
#     if value < 0: return 0
#     return value

# def increase_contrast(img, contr):
#     img = img.copy()
#     factor = (259 * (contr + 255)) / (255 * (259 - contr))
#     for i, row in enumerate(img):
#         for j, col in enumerate(row):
#             tmp = img[i][j]
#             img[i][j] = trunc(int(factor * (col - 128) + 128))
#             #print(str(tmp) + " -> " + str(img[i][j]))
#     return img

# def crop_circles(img):
#     img = img.copy()
#     height,width = img.shape
#     mask = np.zeros((height,width), np.uint8)
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 1000, param1 = 50, param2 = 30, minRadius = 0, maxRadius = 0)
#     # for i in circles[0,:]:
#     #     i[2]=i[2]+4
#     #     # Draw on mask
#     #     cv2.circle(mask,(i[0],i[1]),i[2],(255,255,255),thickness=-1)
#     # cv2.bitwise_and(img, img, mask=mask)
#     return img
#
# def find_circles_and_draw(img, back):
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 2, param1 = 30, param2 = 15, minRadius = 1, maxRadius = 70)
#     circles = np.uint16(np.around(circles))
#     for i in circles[0,:]:
#         # draw the outer circle
#         cv2.circle(back,(i[0],i[1]),i[2],(0,255,0),2)
#         # draw the center of the circle
#         cv2.circle(back,(i[0],i[1]),2,(0,0,255),3)
#     return back



def find_blobs(img, back):
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 100


    # Filter by Area.
    params.filterByArea = True
    params.minArea = 30

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.75
    params.maxCircularity = 1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.95
    params.maxConvexity = 1

    # Filter by Inertia
    params.filterByInertia = True
    params.maxInertiaRatio = 1
    params.minInertiaRatio = 0.5


    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(back)
    im_with_keypoints = cv2.drawKeypoints(back, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return im_with_keypoints


# def laplacian_and_thresh(image):
#     laplacian = cv2.Laplacian(image,cv2.CV_64F)
#     absolute = np.absolute(laplacian)
#     conv = np.uint8(absolute)
#     thresh = cv2.adaptiveThreshold(conv,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#     return np.uint8(thresh)

dirname = os.path.dirname(__file__)
samples_path = os.path.join(dirname, 'samples')

image_path = os.path.join(samples_path, dirname, 'samples', on_table)

nat = imread(image_path)

nat_resized = resize(nat, (0,0), fx=0.3, fy=0.3)

img = imread(image_path, 0)

img_gray = imread(image_path, IMREAD_GRAYSCALE)

img_resized = resize(img_gray, (0,0), fx=0.3, fy=0.3)

# laplacian = laplacian_and_thresh(img)

kernel = np.ones((5,5),np.uint8)
#
# alpha = 3
# beta = 50
# o1 = addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)





# contrast = increase_contrast(gray, 128)
#
# cuted_circles = crop_circles(contrast)


#gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)


contrast2 = convertScaleAbs(img_resized ,alpha =  2.2,beta = 50)

_, thsd = threshold(contrast2 ,200,255,THRESH_BINARY)

element = getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

tup = 5
o2 = medianBlur(thsd,tup)
#
o3 = dilate(o2,element,iterations = 1)
#
o4 = erode(o3,element,iterations = 1)
#
_, o5 = threshold(o4 ,0,255,THRESH_BINARY + THRESH_OTSU)

o5 = medianBlur(o5,tup)
#
_, cnt, _ = findContours(o5, RETR_TREE, CHAIN_APPROX_NONE)

dotts = find_blobs(o5, nat_resized)
#
# (_, contours, _) = cv2.findContours(o5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#
#drawContours(nat_resized, cnt, -1, (0,255,0), 5)
#
# numberofDice = 2
# label = np.ones((len(cnt), 1), np.uint8)
# label = np.ones((numberofDice, 1), np.uint8)


# show_image("Orig", img_resized)
# show_image("GrayScale", gray)
# show_image("contrast", contrast)
# show_image("cut_circles", cuted_circles)
show_image("contrast2", contrast2)
show_image("thsd", thsd)
show_image("blur", o2)
show_image("Dilate", o3)
show_image("Erose", o4)
show_image("Thresh", o5)
show_image("Kontury", dotts)

waitKey(0)
destroyAllWindows()
