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

random.seed(1000)

def get_imgs(read_folder, actors, num_photos):
    imgs = np.empty([0, 1024])
    label = zeros([0, len(actors)])
    labels = np.empty([0, len(actors)])
    filename_to_img = []
    label_i = -1
    for act in actors:
        label_i+= 1
        i = 0
        for filename in os.listdir(read_folder):
            if act in filename:
                try:
                    img = imread(os.path.join(read_folder,filename), flatten=True)
                    if img is not None:
                        imgs = vstack((imgs, reshape(np.ndarray.flatten(img), [1, 1024])))
                        filename_to_img.append(filename) 
                        i +=1
                        curr_label = zeros([1, len(actors)])
                        curr_label[0][label_i] = 1
                        labels = vstack((labels, curr_label))
                        if i >= num_photos:
                            break
                except Exception as e:
                    print(filename, str(e))
    return imgs, labels, filename_to_img


def f(x, y, theta):
    #x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    #x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)

def grad_descent(f, df, x, y, init_t, alpha):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 30000
    iter  = 0 
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= (alpha*df(x, y, t).reshape(1024, 1))
        if iter % 500 == 0:
            print "Iter", iter
            print "df = ", alpha*df(x, y, t)
            print "t = ", t 
            print "Gradient: ", df(x, y, t), "\n"
        iter += 1
    return t

def get_sets(x, y, actors, filename_to_img, training_size=100, validation_size=10, test_size=10):
    total_size = training_size + validation_size + test_size
    
    training_set = []
    training_y = []
    validation_set = []
    validation_y = []
    test_set = []
    test_y = []
    
    for actor_i in range(len(actors)):
        
        # get random indices
        img_i = range(actor_i*total_size, actor_i*total_size + total_size)
        random.shuffle(img_i)
        randomized_set = [x[j] for j in img_i]
        
        training_set.extend(randomized_set[:training_size])
        validation_set.extend(randomized_set[training_size: training_size + validation_size])
        test_set.extend(randomized_set[training_size + validation_size:])
        
        training_y.extend(y[actor_i*total_size : actor_i*total_size+training_size])
        validation_y.extend(y[actor_i*total_size+training_size : actor_i*total_size+training_size+validation_size])
        test_y.extend(y[actor_i*total_size+training_size+validation_size : (actor_i+1)*(total_size)])
    
    return np.array(training_set), np.array(training_y), np.array(validation_set), np.array(validation_y), np.array(test_set), np.array(test_y)

def linear_classifier():
    TRAINING_SIZE = 2
    VAL_SIZE = 10
    TEST_SIZE = 10
    TOTAL_SIZE = TRAINING_SIZE + VAL_SIZE + TEST_SIZE
    #training set
    actors = ["carell", "hader"]
    x, y, filename_to_img = get_imgs("filtered_male/", actors, TOTAL_SIZE)
    x /=255.
    
    training_set, training_y, validation_set, validation_y, test_set, test_y = get_sets(x, y, actors, filename_to_img, TRAINING_SIZE, VAL_SIZE, TEST_SIZE)
    
    #x = vstack(ones([1, 200]))
    theta = np.random.rand(1024, 1)*(1E-12)
    
    # linear classifier y has one column
    training_y = reshape(training_y.T[1], (1, TRAINING_SIZE*len(actors)))
    validation_y = reshape(validation_y.T[1], (1, VAL_SIZE*len(actors)))
    test_y = reshape(test_y.T[1], (1, TEST_SIZE*len(actors)))
    
    t = grad_descent(f, df, training_set.T, training_y, theta, 1E-7)
    
    # imshow(reshape(t[1:], [32, 32]))
    # show()
    # imsave('theta.png', reshape(t[1:], [32,32]))
    
    imshow(reshape(t, [32, 32]))
    show()
    imsave('theta.png', reshape(t, [32,32]))
    # t = imread("theta.png", True)
    # t = reshape(t, [1024, 1])
    
    print("theta: ", t)
    return t


def part3():
    t = linear_classifier()
    
    train_p = performance(training_set.T, training_y, t, TRAINING_SIZE*len(actors))
    val_p = performance(validation_set.T, validation_y, t, VAL_SIZE*len(actors))
    test_p = performance(test_set.T, test_y, t, TEST_SIZE*len(actors))

    print("TRAIN PERFORMANCE: %f", train_p*100)
    print("VALIDATION PERFORMANCE: %f", val_p*100)
    print("TEST PERFORMANCE: %f", test_p*100)
    
def performance(x, y, theta, size):
    correct_y = y[0]
    # x = vstack( (ones((1, x.shape[1])), x))
    h = dot(theta.T, x)
    predicted_y = []
    
    for i in range(len(correct_y)):
        if h[:,i] > 0.5:
            predicted_y.append(1)
        else:
            predicted_y.append(0)
        
    num_correct = 0
    for c, p in zip(correct_y, predicted_y):
        if p == c:
            num_correct +=1
        
    return num_correct/float(size)
    
linear_classifier()
    
    
    
    
    
    
    
    
    
    
    
    
    