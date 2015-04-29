import numpy as np
from selection.covtest import covtest

def parameters(n, rho, pos):
    """
    design matrix and mu vector
    """

    cov = ((1-rho) * np.identity(n) + rho * np.ones((n,n)))
    X = np.linalg.cholesky(cov).T
    beta = np.zeros(n)
    beta[pos] = 1.
    mu = np.dot(X, beta)
    return X, mu, beta

def constraints(X, pos):
    n, p = X.shape
    while True:
        Y = np.random.standard_normal(n)
        con, _, idx, sign = covtest(X, Y, sigma=1)
        if idx == pos and sign == +1:
            initial = Y.copy()
            break
    return con, initial
