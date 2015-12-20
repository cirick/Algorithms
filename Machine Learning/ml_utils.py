import numpy as np

import matplotlib.pyplot as plt
import seaborn; seaborn.set()  # plot styling

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def mapfeat(x1, x2):
    degree = 6
    x1 = np.matrix(x1)
    x2 = np.matrix(x2)
    out = np.ones((x1.size,1))
    idx = 1
    for i in range(1,degree+1):
        for j in range(0,i+1):
            #x = np.array(x1**(i-j) * x2**j)
            x = np.multiply(np.power(x1,i-j),np.power(x2,j))
            out = np.hstack([out, x])
            idx += 1
    return out

def normalize(x):
    mu = np.mean(x)
    sigma = np.std(x)
    xnorm = (x - mu) / sigma
    return mu, sigma, xnorm

def plot_dec_boundary(theta, x, y, mapFeat=False):
    theta = np.matrix(theta).T
    if mapFeat is True:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
    else:
        u = np.linspace(np.min(x[:,1]), np.max(x[:,1]), 50)
        v = np.linspace(np.min(x[:,2]), np.max(x[:,2]), 50)

    z = np.zeros((u.size, v.size))

    #% Evaluate z = theta*x over the grid
    for i in range(0,u.size):
        for j in range(0,v.size):
            if mapFeat is True:
                x_m = mapfeat(u[i], v[j]) 
            else:
                #x_m = np.matrix([1,u[i],v[j],u[i]**2,u[i]**4])
                x_m = np.matrix([1,u[i],v[j],u[i]**2])
            z[i,j] = x_m * theta
    z = z.T # important to transpose z before calling contour
    print z, z.shape

    # Plot z = 0
    # Notice you need to specify the range [0, 0]
    plt.figure()
    #plt.scatter(x[:, 1], x[:, 2], c=y, cmap=plt.cm.Paired)
    plt.scatter(np.extract(y==1, x[:, 1]), np.extract(y==1, x[:, 2]), c='b', marker='o', label='admitted')
    plt.scatter(np.extract(y==0, x[:, 1]), np.extract(y==0, x[:, 2]), c='r', marker='o', label='declined')
    #ax = fig.add_subplot(111)
    plt.xlabel('Test 1 scores')
    plt.ylabel('Test 2 scores')
    plt.legend()
    plt.contour(u, v, z, [0,0], linewidth = 2, cmap=plt.cm.Paired)
    plt.show()