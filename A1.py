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

act = list(set([a.split("\t")[0] for a in open("facescrub_actors.txt").readlines()]))


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

testfile = urllib.URLopener()            


#Note: you need to create the uncropped folder first in order 
#for this to work

for a in act:
    name = a.split()[1].lower()
    i = 0
    for line in open("facescrub_actors.txt"):
        if a in line:
            crop_coords = line.split()[5].split(",")
            crop_coords = tuple(map(int, crop_coords))
            filename = name+str(i)+"-"+str(crop_coords)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            try:
                testfile.retrieve(line.split()[4], "uncropped/"+filename)
            except Exception as e:
                print(str(e))
            #timeout is used to stop downloading images which take too long to download
            #timeout(testfile.retrieve, (line.split()[4], "uncropped/"+filename), {}, 30)
            if not os.path.isfile("uncropped/"+filename):
                continue

            
            print filename
            i += 1
    
    