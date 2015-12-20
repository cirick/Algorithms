import numpy as np

def cost(theta, x, y):
    m = y.size
    hypothesis = np.dot(x, theta)
    square_error = (hypothesis - y) ** 2
    return 1.0/(2.0*m) * np.sum(square_error)


def gradient(theta, x, y, alpha, iterations):
    m = y.size
    n = x[0].size
    d = np.zeros(shape=(n, 1))
    j_hist = np.zeros(shape=(iterations, 1))

    for i in range(iterations):
        hypothesis = np.dot(x, theta)

        loss = hypothesis - y
        for j in range(n):
            d[j] = (1.0 / m) * np.sum(loss.T * x[:, j])

        for j in range(n):
            theta[j] = theta[j] - alpha * d[j]

        j_hist[i, 0] = cost(theta, x, y)

    return theta, j_hist



