from cv2 import *
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
import os


on_paper = "IMG_20181118_220206.jpg"
on_table = "IMG_20181118_220153.jpg"


class Dice:
    def __init__(self, point):
        self.dotts.append(point)

def show_image(title, image):
    namedWindow(title, WINDOW_NORMAL)
    imshow(title, image)


def find_blobs(img, back):
    params = cv2.SimpleBlobDetector_Params()

    #Change thresholds
    # params.minThreshold = 0
    # params.maxThreshold = 20

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 2000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.7
    params.maxCircularity = 1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.6
    params.maxConvexity = 1

    # Filter by Inertia
    params.filterByInertia = True
    params.maxInertiaRatio = 1
    params.minInertiaRatio = 0.6


    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(img)
    im_with_keypoints = cv2.drawKeypoints(back, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    print(len(keypoints))
    for k in keypoints:
        im_with_keypoints = cv2.circle(back, (np.int(k.pt[0]),np.int(k.pt[1])), radius=np.int(k.size*0.8), color=(0,255,0), thickness=-1)
    return im_with_keypoints


def read_image(name):
    dirname = os.path.dirname(__file__)
    samples_path = os.path.join(dirname, 'samples')
    image_path = os.path.join(samples_path, dirname, 'samples', name)
    nat = imread(image_path)
    img_gray = imread(image_path, IMREAD_GRAYSCALE)

#return resize(img_gray, (0,0), fx=0.3, fy=0.3), resize(nat, (0,0), fx=0.3, fy=0.3)
    return img_gray, nat

def filter_image(img):
    contrast = convertScaleAbs(img ,alpha =  2.2,beta = 50)
    _, thresh = threshold(contrast ,254,255,THRESH_BINARY)
    element = getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    o2 = medianBlur(thresh,5)
    o3 = dilate(o2,element,iterations = 1)
    o4 = erode(o3,element,iterations = 1)
    # _, cnt, _ = findContours(o4, RETR_TREE, CHAIN_APPROX_NONE)
    return o4




def count_dotts(file_name):
    img_resized, nat_resized = read_image(file_name)

    o4 = filter_image(img_resized)
    blobs = find_blobs(o4, nat_resized.copy())

    neg = 255 - o4
    blobs_neg = find_blobs(neg, blobs)

    imwrite(os.path.join('results', file_name), blobs_neg)
    # show_image("Kontury", blobs_neg)
    # show_image("thresh", o4)
    # show_image("neg", neg)

    waitKey(0)
    destroyAllWindows()

dirname = os.path.dirname(__file__)
samples_path = os.path.join(dirname, 'samples')
for (dirpath, dirnames, filenames) in os.walk(samples_path):
    for name in filenames:
        count_dotts(name)
        print(name)
    break
# count_dotts(on_table)
