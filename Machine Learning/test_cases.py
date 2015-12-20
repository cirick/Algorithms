import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # plot styling

import scipy.optimize as opt
from scipy.io import loadmat

import linear_regression as linreg
import logistic_regression as logreg
import ml_utils as util

def test_linreg_single_var():
    data = pd.read_csv('data/ex1data1.txt', header=None)
    n = data[0].size

    x0 = np.ones((n, 1))
    x1 = data[0]
    x1 = x1.reshape(n, 1)
    x = np.hstack([x0, x1])

    y = data[1]
    y = y.reshape(n, 1)

    theta = np.zeros((2, 1))

    iterations = 1500
    alpha = 0.01
    theta, j_history = linreg.gradient(theta, x, y, alpha, iterations)
    print theta

    print '35,000 people = %f' % (np.array([1,3.5]).dot(theta).flatten() * 10000)
    print '70,000 people = %f' % (np.array([1,7]).dot(theta).flatten() * 10000)


def test_linreg_multi_var():
    data = pd.read_csv('data/ex1data2.txt', header=None)
    n = data[0].size

    x0 = np.ones((n, 1))
    x1 = data[0]
    x1 = x1.reshape(n, 1)
    x1u, x1s, x1 = util.normalize(x1)
    x2 = data[1]
    x2 = x2.reshape(n, 1)
    x2u, x2s, x2 = util.normalize(x2)
    x = np.hstack([x0, x1, x2])

    y = data[2]
    y = y.reshape(n, 1)

    theta = np.zeros((3, 1))

    iterations = 10000
    alpha = 0.001
    theta, j_history = linreg.gradient(theta, x, y, alpha, iterations)
    print theta
    '''
    plt.plot(np.arange(iterations), j_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.show()'''

    print '2000sqft + 3bedrooms = %f' % np.array([1,(2000-x1u)/x1s,(3-x2u)/x2s]).dot(theta).flatten()
    print '4000sqft + 5bedrooms = %f' % np.array([1,(4000-x1u)/x1s,(5-x2u)/x2s]).dot(theta).flatten()
    print '4000sqft + 6bedrooms = %f' % np.array([1,(4000-x1u)/x1s,(6-x2u)/x2s]).dot(theta).flatten()
    print '2000sqft + 6bedrooms = %f' % np.array([1,(2000-x1u)/x1s,(6-x2u)/x2s]).dot(theta).flatten()


def test_linreg_compare():
    data = pd.read_csv('data/Advertising.csv', index_col=0)
    x1 = data.values[:,0]
    y = data.values[:,3]

    y = y.reshape(y.size,1)
    x1 = x1.reshape(x1.size,1)
    x1u, x1s, x1 = util.normalize(x1)
    x0 = np.ones((x1.size,1))
    x = np.hstack([x0,x1])

    theta = np.zeros((2,1))

    iterations = 1500
    alpha = 0.01
    theta, j_history = linreg.gradient(theta, x, y, alpha, iterations)
    print theta

    #plt.plot(np.arange(iterations), J_history)
    #plt.xlabel('Iterations')
    #plt.ylabel('Cost Function')
    #plt.show()

    print '$50000 TV dollars = %f widgets' % (np.array([1,(50-x1u)/x1s]).dot(theta).flatten()*1000)
    #print '70,000 people = %f' % (np.array([1,7]).dot(theta).flatten() * 10000)


def test_linreg_multi_compare():
    data = pd.read_csv('data/Advertising.csv', index_col=0)
    x1 = data.values[:,0]
    x2 = data.values[:,1]
    x3 = data.values[:,2]
    y = data.values[:,3]

    y = y.reshape(y.size,1)
    x1 = x1.reshape(x1.size,1)
    x1u, x1s, x1 = util.normalize(x1)
    x2 = x2.reshape(x2.size,1)
    x2u, x2s, x2 = util.normalize(x2)
    x3 = x3.reshape(x3.size,1)
    x3u, x3s, x3 = util.normalize(x3)
    x0 = np.ones((x1.size,1))
    x = np.hstack([x0,x1,x2,x3])

    theta = np.zeros((4,1))

    iterations = 1500
    alpha = 0.01
    theta, j_history = linreg.gradient(theta, x, y, alpha, iterations)
    print theta

    #plt.plot(np.arange(iterations), J_history)
    #plt.xlabel('Iterations')
    #plt.ylabel('Cost Function')
    #plt.show()

    print '$100k/$25k/$25k TV/Radio/Paper = %f widgets' % (np.array([1,(100-x1u)/x1s,(25-x2u)/x2s,(25-x3u)/x3s]).dot(theta).flatten()*1000)
    #print '70,000 people = %f' % (np.array([1,7]).dot(theta).flatten() * 10000)

def test_logreg_accept_scores():
    data = pd.read_csv('data/ex2data1.txt', header=None)
    x1 = data[0]
    x2 = data[1]
    x1 = x1.reshape(x1.size, 1)
    x2 = x2.reshape(x2.size, 1)
    x3 = x1 ** 2
    #x4 = x1 ** 4

    x0 = np.ones((data[0].size, 1))
    x = np.hstack([x0, x1, x2, x3])
    #x = np.hstack([x0, x1, x2, x3, x4])

    y = data[2]
    y = y.reshape(y.size, 1)
    l = 0.0025 #squared term

    #theta must be an array to work with fmin_tnc x0 param
    theta = np.zeros(x[0].size)

    print 'initial cost %f' % logreg.cost(theta, x, y, l)
    
    #must take an array
    #param, neval, status = opt.fmin_tnc(func=cost, x0=theta, fprime=gradient, args=(x, y), approx_grad=True, epsilon=0.000000000001)
    param, neval, status = opt.fmin_tnc(func=logreg.cost, x0=theta, fprime=logreg.gradient, args=(x, y, l), maxfun=1000)
    print param
    print 'Neval %d status %d\n' % (neval, status)
    print 'Cost: %f' % logreg.cost(param, x, y, 0)
    #print 'COE: %f' % sigmoid(np.dot(np.array([1,45,85, (45**2), (45**4)]), param.T))
    #print 'COE: %f' % sigmoid(np.dot(np.array([1,75.3,46.3, (75.3**2), (75.3**4)]), param.T))
    #print 'COE: %f' % sigmoid(np.dot(np.array([1,82.2,41.9, (82.2**2), (82.2**4)]), param.T))
    print 'COE: %f' % util.sigmoid(np.dot(np.array([1,45,85, 45**2]), param.T))
    print 'COE: %f' % util.sigmoid(np.dot(np.array([1,75.3,46.3, 75.3**2]), param.T))
    print 'COE: %f' % util.sigmoid(np.dot(np.array([1,82.2,41.9, 82.2**2]), param.T))

    p = logreg.predict(param,x)
    #compare predictions from trained params with original truth
    print 'accuracy %f' % (np.sum(np.logical_not(np.logical_xor(p,data[2]))) / float(y.size))

    util.plot_dec_boundary(param, x, y, False)
    
    '''
    # gradient descent needs normalization with large manitude values, otherwise it's very slow to converge 
    iterations = 1500000
    alpha = 0.001
    #alpha = 0.1
    theta, j_history = gradient_d(theta, x, y, alpha, iterations)
    print 'Cost: %f\n' % cost(theta, x, y)
    #print 'COE: %f\n' % sigmoid(np.dot(np.array([1,(45-x1u)/x1s,(85-x2u)/x2s]), theta))
    print 'COE: %f\n' % sigmoid(np.dot(np.array([1,45,85]), theta.T))

    plt.plot(np.arange(iterations), j_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost Function')
    plt.show()
    '''


def test_logreg_engines_reg():
    data = pd.read_csv('data/ex2data2.txt', header=None)
    x1 = data[0]
    x2 = data[1]
    x1 = x1.reshape(x1.size, 1)
    x2 = x2.reshape(x2.size, 1)

    y = data[2]
    y = y.reshape(y.size, 1)

    x = util.mapfeat(x1,x2)
    theta = np.zeros(x.shape[1])
    l = 1.0

    print 'initial cost %f' % logreg.cost(theta, x, y, l)
    print logreg.gradient(theta,x,y,l).shape

    param, neval, status = opt.fmin_tnc(func=logreg.cost, x0=theta, fprime=logreg.gradient, args=(x, y, l))
    print param
    print 'Neval %d status %d\n' % (neval, status)
    print 'Cost: %f' % logreg.cost(param, x, y, l)
    
    p = logreg.predict(param,x)
    #compare predictions from trained params with original truth
    print 'accuracy %f' % (np.sum(np.logical_not(np.logical_xor(p,data[2]))) / float(y.size))
    util.plot_dec_boundary(param,x,y,True)

def test_logreg_hand_written():
    data = loadmat('data/ex3data1.mat')


    # len() gives number of rows

    #x0 = np.ones((data['X'].shape[0], 1))
    #x = np.hstack([x0, data['X']])
    x = data['X']
    y = data['y']

    l = 0.1
    all_theta = logreg.one_vs_all(x,y,10,l)
    p = logreg.one_vs_all_predict(all_theta,x)
    #all_theta = one_vs_all_comp(x,y,10,l)
    #p = one_vs_all_predict_comp(all_theta,x)

    correct = [1 if a == b else 0 for (a, b) in zip(p, data['y'])]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'accuracy = {0}%'.format(accuracy * 100)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan, linewidth=250)

    test_linreg_single_var()
    test_linreg_multi_var()
    test_linreg_compare()
    test_linreg_multi_compare()

        
    test_logreg_accept_scores()
    test_logreg_engines_reg()
    test_logreg_hand_written()
