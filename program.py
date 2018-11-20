# import numpy as np
import os
from math import sqrt
# import matplotlib.pyplot as plt
#
# from skimage import measure, data, io, filters, morphology, feature

import numpy as np
import matplotlib.pyplot as plt

from skimage import data, color, filters, feature, morphology
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte

class Circle(object):
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

    def overlap(self, x, y, r):
        c = sqrt(pow(abs(x - self.x),2) + pow(abs(y - self.y),2))
        if c > r + self.r:
           return False
        return True

def count_pips(image):
    # http://scikit-image.org/docs/dev/auto_examples/filters/plot_hysteresis.html#sphx-glr-auto-examples-filters-plot-hysteresis-py

    edges = filters.sobel(image)

    low = 0.03
    high = 0.35

    lowt = (edges > low).astype(int)
    hight = (edges > high).astype(int)
    hyst = filters.apply_hysteresis_threshold(edges, low, high)

    edges = filters.median(lowt, morphology.disk(10))

    # http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html

    hough_radii = np.arange(20, 40, 5)
    hough_res = hough_circle(edges, hough_radii)

    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii)
    #print(accums)
    #print(len(radii))


    image = color.gray2rgb(image)

    counted = []
    for center_y, center_x, radius, acc in zip(cy, cx, radii, accums):
        if acc < 0.9:
            break
        for circ in counted:
            if circ.overlap(center_y, center_x, radius):
                break
        else:
            counted.append(Circle(center_y, center_x, radius))
            circy, circx = circle_perimeter(center_y, center_x, radius)
            try:
                image[circy, circx] = (1, 0, 0)
            except IndexError:
                print(circx)
                print(circy)

    #print(len(counted))
    plt.imshow(image, cmap=plt.cm.gray)
    plt.axis('off')
    plt.title(len(counted))
    plt.savefig(os.path.join('results', name))
    #plt.show()

# read all sample images
dirname = os.path.dirname(__file__)
samples_path = os.path.join(dirname, 'samples')
for (dirpath, dirnames, filenames) in os.walk(samples_path):
    for name in filenames:
        image_path = os.path.join(samples_path, dirname, 'samples', name)
        image = data.load(image_path, as_gray=True)
        count_pips(image)
        print(name)
    break
