import pandas as pd
import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat
import ml_utils as util

def cost(theta, x, y, l):
    m = x.shape[0]
    theta = np.matrix(theta) #1xn matrix
    x = np.matrix(x) #force to numpy matrix type
    y = np.matrix(y)

    hypothesis = util.sigmoid(x * theta.T) #matrix multiply for '*' when both are matrices 
    pos = np.multiply(-y, np.log(hypothesis)) #np.multiply to force element-wise with matrix types
    neg = np.multiply((1-y), np.log(1-hypothesis))
    reg = 0
    if l > 0:
    	reg = (l / (2.0 * m)) * np.sum(np.power(theta[:,1:theta.shape[1]],2))

    return (np.sum(pos - neg) / m) + reg

def gradient(theta, x, y, l):
	m = len(x)
	n = len(x[0])
	theta = np.matrix(theta).T #nx1 matrix
	x = np.matrix(x) #force to numpy matrix type
	y = np.matrix(y)

	hypothesis = util.sigmoid(x * theta)
	loss = hypothesis - y
	grad = x.T*(loss) / m
	reg = grad
	# regularize except intercept grad
	if l > 0:
		reg = grad + ((l * theta) / m )
		reg[0,0] -= ((l * theta[0,0]) / m )

	#scipy optimize needs flat array returned for the gradient
	return np.array(reg).ravel()

def predict(theta, x):
    theta = np.matrix(theta) #1xn matrix
    x = np.matrix(x) #force to numpy matrix type
    hypothesis = util.sigmoid(x * theta.T)

    return [1 if i >= 0.5 else 0 for i in hypothesis]

def one_vs_all(x, y, num_labels, l):
	m = x.shape[0]
	n = x.shape[1]

	all_theta = np.zeros((num_labels, n + 1))
	x = np.insert(x, 0, values=np.ones(m), axis=1)

	for k in range(1, num_labels+1):
		theta = np.zeros(n + 1)

		yi = np.array([1 if label == k else 0 for label in y])
		yi = np.reshape(yi, (m, 1))

		#CG performs the same as TNC for this dataset and TNC is faster
		fmin = opt.minimize(fun=cost, x0=theta, args=(x, yi, l), method='TNC', jac=gradient)
		print fmin.nit, fmin.fun

		all_theta[k-1,:] = fmin.x

	return all_theta

def one_vs_all_predict(all_theta, x):
    m = x.shape[0]
    n = x.shape[1]
    
    # same as before, insert ones to match the shape
    x = np.insert(x, 0, values=np.ones(m), axis=1)
    
    # convert to matrices
    x = np.matrix(x)
    all_theta = np.matrix(all_theta)
    
    # compute the class probability for each class on each training instance
    h = util.sigmoid(x * all_theta.T)
    h_argmax = np.argmax(h, axis=1)
    h_argmax += 1

    return h_argmax

