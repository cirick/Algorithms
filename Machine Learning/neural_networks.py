import numpy as np
import scipy.optimize as opt
from scipy.io import loadmat

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def init_weights_rand(L_in, L_out):
    epsilon_init = 0.12
    return np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init

def forward_prop(x, theta1, theta2):
    m = x.shape[0]

    a1 = np.insert(x, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = sigmoid(z2)

    a2 = np.insert(a2, 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3) 

    return a1, z2, a2, z3, h

def nn_cost(params, il_size, hl_size, k, x, y, l):
    theta1 = np.matrix(np.reshape(params[: hl_size * (il_size + 1)], ( hl_size, (il_size + 1))))
    theta2 = np.matrix(np.reshape(params[hl_size * (il_size + 1) :], ( k, (hl_size + 1))))
    x = np.matrix(x)
    y = np.matrix(y)
    m = x.shape[0]

    #hypothesis
    a1, z2, a2, z3, h = forward_prop(x, theta1, theta2)

    #cost
    pos = np.multiply(-y, np.log(h))
    neg = np.multiply((1-y), np.log(1-h))
    cost = np.sum(pos - neg) / m

    #cost regularization
    reg = 0
    if l > 0:
        t1r = np.sum(np.power(theta1[:,1:theta1.shape[1]],2))
        t2r = np.sum(np.power(theta2[:,1:theta2.shape[1]],2))
        reg = (l / (2.0 * m)) * (t1r + t2r)
    cost = cost + reg

    #graident
    e3 = h - y #5000 x 10
    z2 = np.insert(z2, 0, values=np.ones(m), axis=1)
    e2 = np.multiply((theta2.T * e3.T).T,sigmoid_grad(z2)) # 5000 x 26

    grad1 = (e2[:,1:(hl_size+1)].T * a1) / m # 25 x 401
    grad2 = (e3.T * a2) / m # 10 x 26

    #gradient regularization
    if l > 0:
        grad1[:,1:] = grad1[:,1:] + (theta1[:,1:] * l) / m
        grad2[:,1:] = grad2[:,1:] + (theta2[:,1:] * l) / m
    grad = np.hstack((np.array(grad1).ravel(), np.array(grad2).ravel()))

    return cost, grad


def test_hand_written():
    data = loadmat('data/ex3data1.mat')
    thetas = loadmat('data/ex4weights.mat')
    x = data['X']
    y = data['y']
    params = np.hstack((thetas['Theta1'].ravel(),thetas['Theta2'].ravel()))
    m = x.shape[0]

    il_size  = 400
    hl_size = 25
    k = 10
    l = 1

    #class labels
    y_k = np.zeros([m,k])
    for i in range(m):
        c = y[i]
        y_k[i] = np.array([1 if label+1 == c else 0 for label in range(k)])

    cost, grad = nn_cost(params, il_size, hl_size, k, x, y_k, l)
    print 'Initial cost %f' % cost

    fmin = opt.minimize(fun=nn_cost, x0=params, args=(il_size, hl_size, k, x, y_k, l), method='CG', jac=True, options={'maxiter': 250})
    print fmin

    theta1p = np.matrix(np.reshape(fmin.x[:hl_size * (il_size + 1)], (hl_size, (il_size + 1))))
    theta2p = np.matrix(np.reshape(fmin.x[hl_size * (il_size + 1):], (k, (hl_size + 1))))

    a1, z2, a2, z3, h = forward_prop(x, theta1p, theta2p)
    y_pred = np.array(np.argmax(h, axis=1) + 1)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    accuracy = (sum(map(int, correct)) / float(len(correct)))
    print 'accuracy = %f' % (accuracy * 100)


if __name__ == '__main__':
    #np.set_printoptions(threshold=np.nan, linewidth=250)
    #test_accept_scores()
    #test_engines_reg()
    test_hand_written()