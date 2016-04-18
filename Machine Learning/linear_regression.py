import numpy as np
import matplotlib.pyplot as plt

from sklearn import linear_model


class LinReg:
    def __init__(self,
                 epochs=1000,
                 alpha=0,
                 fit_intercept=True,
                 normalize=False):
        self.epochs = epochs
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.theta = None
        self.coef_ = None
        self.intercept_ = None
        self.j_hist = None

        self.m = 0
        self.n = 0

        self.mu_ = list()
        self.sigma_ = list()
        self.norm_ = np.array([])

    def cost(self, theta, x, y):
        m = y.size
        hypothesis = np.dot(x, theta)
        square_error = (hypothesis - y) ** 2
        return 1.0 / (2.0 * m) * np.sum(square_error)

    # X: [n_samples, n_features]
    # Y: [n_samples, n_targets]
    def fit(self, x, y):
        self.m = y.size
        self.n = x[0].size + 1
        d = np.zeros((self.n, 1))

        if self.fit_intercept:
            b = np.ones((self.m, 1))
        else:
            b = np.zeros((self.m, 1))

        if self.normalize:
            self.normalize_feat(x)
        else:
            self.norm_ = x

        X = np.hstack([b, self.norm_])

        self.theta = np.zeros((self.n, 1))
        self.j_hist = np.zeros((self.epochs, 1))

        for i in range(self.epochs):
            hypothesis = np.dot(X, self.theta)

            loss = np.subtract(hypothesis, y)
            d = (1.0 / self.m) * np.dot(X.T, loss)
            self.theta = np.subtract(self.theta, np.multiply(self.alpha, d))

            self.j_hist[i, 0] = self.cost(self.theta, X, y)

        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:self.n]

    def normalize_feat(self, x):
        xnorm = np.array([]).reshape(self.m, 0)

        for i in range(self.n - 1):
            feat = x[:, i]
            mu = np.mean(feat)
            sigma = np.std(feat)
            cnorm = (feat - mu) / sigma
            xnorm = np.concatenate((xnorm, cnorm[:, np.newaxis]), axis=1)

            self.mu_.append(mu)
            self.sigma_.append(sigma)

        self.norm_ = xnorm

    def predict(self, x):
        m = x.shape[0]
        if self.fit_intercept:
            b = np.ones((m, 1))
        else:
            b = np.zeros((m, 1))

        w = np.array([]).reshape(m, 0)
        if self.normalize:
            for i in range(0, self.n - 1):
                t = np.subtract(x[:, i, np.newaxis], np.full((m, 1), self.mu_[i]))
                t = np.divide(t, np.full((m, 1), self.sigma_[i]))

                w = np.concatenate((w, t), axis=1)
        else:
            w = x

        X = np.concatenate((b, w), axis=1)
        return X.dot(self.theta)

    #@todo: score() method


def test_linreg(mlr, file, test_set):
    data = np.genfromtxt(file, delimiter=',')

    m, n = data.shape
    X = data[:, 0:n - 1]
    y = data[:, n - 1, np.newaxis]

    mlr.fit(X, y)

    clf = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    clf.fit(X, y)

    t = mlr.predict(test_set)
    t1 = clf.predict(test_set)
    err = max(t1 - t)
    if err < 0.1:
        print 'Pass ' + str(err)
    else:
        print 'Fail ' + str(err)

    plt.figure()
    plt.plot(mlr.j_hist)
    plt.show()

def test_single_compare():
    mlr = LinReg(epochs=10000, alpha=0.01, fit_intercept=True, normalize=True)
    test_set = np.array([[3.5],
                         [7]])
    test_linreg(mlr, 'data/ex1data1.txt', test_set)


def test_multi_compare():
    mlr = LinReg(epochs=10000, alpha=0.01, fit_intercept=True, normalize=True)
    test_set = np.array([[2000, 4],
                         [4000, 5],
                         [4000, 6],
                         [2000, 2]])
    test_linreg(mlr, 'data/ex1data2.txt', test_set)


def test_single_ad():
    mlr = LinReg(epochs=10000, alpha=0.01, fit_intercept=True, normalize=True)
    test_set = np.array([[230, 37, 69],
                         [40, 40, 40],
                         [100, 10, 40],
                         [10, 250, 10]])
    test_linreg(mlr, 'data/Advertising.csv', test_set)

if __name__ == '__main__':
    test_single_compare()
    test_multi_compare()
    test_single_ad()
