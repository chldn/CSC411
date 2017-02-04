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

random.seed(20) #20 - 71%

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
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)

def check_grad_multiclass(x, y, theta, coords):
    for coord in coords:
        h_r_c = 0.000001
        h = np.zeros([1025, 6])
        h[coord[0],coord[1]] = h_r_c
        x = x.T.reshape(1024, 1)
        y = y.reshape(1,6)
        print "Validating gradient function at: ", coords
        print "\t Finite Difference=", (f_multiclass(x, y, theta+h) - f_multiclass(x, y, theta-h))/(2*h_r_c)
        print "\t df[{},{}]= {}".format(coord[0], coord[1], df_multiclass(x, y, theta)[coord[0], coord[1]])

def f_multiclass(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum(np.square((y - dot(theta.T,x).T)))

def df_multiclass(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return 2*dot(x, (dot(theta.T,x)-y.T).T)

def grad_descent(f, df, x, y, init_t, alpha, dim_row, dim_col, multiclass):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = 3000 #30000
    iter  = 0 
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        if multiclass:
            t -= alpha*df_multiclass(x, y, t)#.reshape(dim_row+1, dim_col)
        else:
            t -= alpha*df(x, y, t).reshape(dim_row+1, dim_col)
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

def performance(x, y, theta, size):
    correct_y = y[0]
    x = vstack( (ones((1, x.shape[1])), x))
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
    
def performance_multiclass(x, y, theta, size):
    correct_y = y
    x = vstack( (ones((1, x.shape[1])), x))
    h = dot(theta.T, x).T
    predicted_y = []
    
    num_correct = 0
    for i in range(len(correct_y)):
        if h[i].argmax() == correct_y[i].argmax():
            num_correct+=1
        
    return num_correct/float(size)
    
    
def linear_classifier(training_set, training_y, dim_row, dim_col, alpha):
    #x = hstack((ones([dim_col, 1]), training_set))
    theta = np.random.rand(dim_row+1, dim_col)*(1E-9)
    t = grad_descent(f, df, training_set.T, training_y, theta, alpha, dim_row, dim_col, multiclass=0)
    
    return t

def linear_classifier_multiclass(training_set, training_y, dim_row, dim_col, alpha):
    #x = hstack((ones([dim_col, 1]), training_set))
    theta = np.random.rand(dim_row+1, dim_col)*(1E-9)
    t = grad_descent(f, df_multiclass, training_set.T, training_y, theta, alpha, dim_row, dim_col, multiclass=1)
    
    return t

def part3():
    TRAINING_SIZE = 100
    VAL_SIZE = 10
    TEST_SIZE = 10
    TOTAL_SIZE = TRAINING_SIZE + VAL_SIZE + TEST_SIZE
    
    # get all images
    actors = ["carell", "hader"]
    x, y, filename_to_img = get_imgs("filtered_male/", actors, TOTAL_SIZE)
    x /=255.
    
    training_set, training_y, validation_set, validation_y, test_set, test_y = get_sets(x, y, actors, filename_to_img, TRAINING_SIZE, VAL_SIZE, TEST_SIZE)
    
    # linear classifier y has one column
    training_y = reshape(training_y.T[1], (1, TRAINING_SIZE*len(actors)))
    validation_y = reshape(validation_y.T[1], (1, VAL_SIZE*len(actors)))
    test_y = reshape(test_y.T[1], (1, TEST_SIZE*len(actors)))
    
    t = linear_classifier(training_set, training_y, 1024, 1, 5E-7)
    print("theta: ", t)
    imshow(reshape(t[1:], [32, 32]), cmap=cm.coolwarm)
    show()
    imsave('part3_theta_training'+str(TRAINING_SIZE)+'.jpg', reshape(t[1:], [32,32]), cmap=cm.coolwarm)
    
    train_p = performance(training_set.T, training_y, t, TRAINING_SIZE*len(actors))
    val_p = performance(validation_set.T, validation_y, t, VAL_SIZE*len(actors))
    test_p = performance(test_set.T, test_y, t, TEST_SIZE*len(actors))
    
    print "f_training= ", f(training_set.T, training_y, t)
    print "f_validation= ", f(validation_set.T, validation_y, t)
    print("TRAIN PERFORMANCE: %f", train_p*100)
    print("VALIDATION PERFORMANCE: %f", val_p*100)
    print("TEST PERFORMANCE: %f", test_p*100)
    

def part5():
    TRAINING_SIZE = 100
    VAL_SIZE = 10
    TEST_SIZE = 10
    OTHER_SIZE = 100
    TOTAL_SIZE = TRAINING_SIZE + VAL_SIZE + TEST_SIZE
    
    # get all images
    actors = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    act_test = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan']
    actors = [a.split()[1].lower() for a in   actors]
    act_test = [a.split()[1].lower() for a in act_test]
    
    # get images for actors
    x, y, filename_to_img = get_imgs("filtered/", actors, TOTAL_SIZE) 
    x /=255.
    y = np.array([[0, 1] if np.nonzero(i)[0][0]<3 else [1, 0] for i in y])
    
    # get images for act_test
    other_x, other_y, filename_to_img = get_imgs("filtered/", act_test, OTHER_SIZE) 
    other_x /=255.
    other_y = np.array([[0, 1] if np.nonzero(i)[0][0]<3 else [1, 0] for i in other_y])
    other_y = reshape(other_y.T[1], (1, OTHER_SIZE*len(act_test)))
    
    training_perf = []
    val_perf = []
    test_perf = []
    other_perf = []
    for TRAINING_SIZE in [1, 5, 10, 50, 100]:
        training_set, training_y, validation_set, validation_y, test_set, test_y = get_sets(x, y, actors, filename_to_img, TRAINING_SIZE, VAL_SIZE, TEST_SIZE)
        
        # linear classifier y has one column
        training_y = reshape(training_y.T[1], (1, TRAINING_SIZE*len(actors)))
        validation_y = reshape(validation_y.T[1], (1, VAL_SIZE*len(actors)))
        test_y = reshape(test_y.T[1], (1, TEST_SIZE*len(actors)))
        
        
        t = linear_classifier(training_set, training_y, 1024, 1, 5E-7)
        # print("theta: ", t)
        # imshow(reshape(t[1:], [32, 32]))
        # show()
        # imsave('part5_theta_'+str(TRAINING_SIZE)+'.png', reshape(t[1:], [32,32]))
        
        train_p = performance(training_set.T, training_y, t, TRAINING_SIZE*len(actors))
        val_p = performance(validation_set.T, validation_y, t, VAL_SIZE*len(actors))
        test_p = performance(test_set.T, test_y, t, TEST_SIZE*len(actors))
        other_p = performance(other_x.T, other_y, t, OTHER_SIZE*len(act_test))
        
        training_perf.append(train_p)
        val_perf.append(val_p)
        #test_perf.append(test_p)
        other_perf.append(other_p)
    
    performances = vstack((np.array(training_perf), np.array(val_perf)))
    #performances = vstack((performances, np.array(test_perf)))
    performances = vstack((performances, np.array(other_perf)))
    
    print("TRAIN PERFORMANCE: ", training_perf)
    print("VALIDATION PERFORMANCE: ", val_perf)
    #print("TEST PERFORMANCE: ", test_perf)
    print("NON TRAINING ACTOR PERFORMANCE: ", other_perf)
    
    part5_plot(performances)
    return performances
    
    
def part5_plot(performance):
    training_data_sizes = [1, 5, 10, 50, 100]
    # performance = part5()
    #performance = array([[ 1.        ,  1.        ,  1.        ,  0.95333333,  0.96166667], [ 0.98333333,  1.        ,  0.98333333,  0.8       ,  0.78333333], [ 1.        ,  0.98333333,  1.        ,  0.85      ,  0.73333333], [ 0.49666667,  0.5       ,  0.5       ,  0.53666667,  0.72166667]])
    plt.plot(training_data_sizes, performance[0], color='k', linewidth=2, marker='o', label="Training Set Performance")
    plt.plot(training_data_sizes, performance[1], color='g', linewidth=2, marker='o', label="Validation Set Performance")
   # plt.plot(training_data_sizes, performance[2], color='b', linewidth=2, marker='o', label="Test Set Performance")
    plt.plot(training_data_sizes, performance[2], color='r', linewidth=2, marker='o', label="Non-Training Set Performance")
    
    plt.title('Training Size vs. Performance for Various Datasets')
    plt.ylim([0,1.2])
    plt.xlabel('Training Size')
    plt.ylabel('Performance')
    plt.legend()
    plt.savefig("part5_plot.jpg")
    plt.show()

def part7():
    TRAINING_SIZE = 100
    VAL_SIZE = 10
    TEST_SIZE = 10
    TOTAL_SIZE = TRAINING_SIZE + VAL_SIZE + TEST_SIZE
    
    # get all images
    actors = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
    actors = [a.split()[1].lower() for a in actors]
    x, y, filename_to_img = get_imgs("filtered/", actors, TOTAL_SIZE)
    x /=255.
    
    training_set, training_y, validation_set, validation_y, test_set, test_y = get_sets(x, y, actors, filename_to_img, TRAINING_SIZE, VAL_SIZE, TEST_SIZE)
    
    t = linear_classifier_multiclass(training_set, training_y, 1024, len(actors), 1.5E-6)
 
    
    # imshow(reshape(t[1:], [32, 32]))
    # show()
    # imsave('theta.png', reshape(t[1:], [32,32]))
    
    # imshow(reshape(t, [32, 32]))
    # show()
    # imsave('theta.png', reshape(t, [32,32]))
    
    for i in range(6):
        imshow(t.T[i][1:].reshape(32,32), cmap=cm.coolwarm)
        imsave('part7_theta_'+str(i), t.T[i][1:].reshape(32,32), cmap=cm.coolwarm)
        show()
    
    train_p = performance_multiclass(training_set.T, training_y, t, TRAINING_SIZE*len(actors))
    val_p = performance_multiclass(validation_set.T, validation_y, t, VAL_SIZE*len(actors))
    test_p = performance_multiclass(test_set.T, test_y, t, TEST_SIZE*len(actors))

    check_grad_multiclass(training_set[1], training_y[1], t, [(2,3), (100,5), (500,5)])
    print("TRAIN PERFORMANCE: %f", train_p*100)
    print("VALIDATION PERFORMANCE: %f", val_p*100)
    print("TEST PERFORMANCE: %f", test_p*100)



part3()
part5()
part7()
    
    
    
    
    
    
    
    
    