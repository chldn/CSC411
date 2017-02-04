# import PySide
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

print("okay got here")

act = list(set([a.split("\t")[0] for a in open("facescrub_actresses.txt").readlines()]))


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result
        
        
def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255.

# import cv2
import os

def filter_imgs(read_folder, out_folder):
    imgs = []
    
    for filename in os.listdir(read_folder):
        try:
            img = imread(os.path.join(read_folder,filename))
            crop_coords = map(int, filename.split("-")[1].split(".")[0][1:-1].split(",")) #format = (x1, y1, x2, y2)
            cropped_img = img[crop_coords[1]:crop_coords[3], crop_coords[0]:crop_coords[2]] # crop format = img[y1:y2, x1:x2]
            filtered_img = imresize(cropped_img, [32, 32])
            
            filtered_img = rgb2gray(filtered_img)
            if filtered_img is not None:
                imgs.append(filtered_img)
                #save to folder
                new_filename = filename.split("-")[0]+".jpg"
                imsave(out_folder+new_filename, filtered_img)
        except Exception as e:
            print(filename, str(e))
    #return imgs

        
#Note: 1. create 'uncropped' folder first
def get_imgs(read_file, out_folder):
    testfile = urllib.URLopener() 
    for a in act:
        name = a.split()[1].lower()
        i = 0
        for line in open(read_file):
            if a in line:
                crop_coords = line.split()[5].split(",")
                crop_coords = tuple(map(int, crop_coords))
                filename = name+str(i)+"-"+str(crop_coords)+'.'+line.split()[4].split('.')[-1]
                #A version without timeout (uncomment in case you need to 
                #unsupress exceptions, which timeout() does)
                try:
                    testfile.retrieve(line.split()[4], out_folder+filename)
                except Exception as e:
                    print(str(e))
                #timeout is used to stop downloading images which take too long to download
                #timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
                if not os.path.isfile(out_folder+filename):
                    continue
    
                
                print filename
                i += 1

# read_file = "facescrub_actresses.txt"
# out_folder = "uncropped_female/"
# get_imgs(read_file, out_folder)

read_folder = "uncropped_female/"
out_folder = "filtered_female/"
filter_imgs(read_folder, out_folder)    
