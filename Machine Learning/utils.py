import numpy as np

def normalize(x):
    mu = np.mean(x)
    sigma = np.std(x)
    xnorm = (x - mu) / sigma
    return mu, sigma, xnorm