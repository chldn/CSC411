from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook 
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

def get_imgs(read_folder):
    imgs = np.empty([0, 1024])
    filename_to_img = []
    for filename in os.listdir(read_folder):
        try:
            img = imread(os.path.join(read_folder,filename), flatten=True)
            if img is not None:
                imgs = vstack((imgs, reshape(np.ndarray.flatten(img), [1, 1024])))
                filename_to_img.append(filename) 

        except Exception as e:
            print(filename, str(e))
    return imgs, filename_to_img

imgs, filename_to_img = get_imgs("filtered_male/")
theta = ones([1, 1024])