import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge

from sklearn.exceptions import NotFittedError

np.random.seed(42)

def gen_random_pair(low, high, n_pairs):
    # method for generating random pair of indices
    # see usage in RFF.estimate_sigma_() method
    n_gend = 0
    
    output = set()
    while(n_gend < n_pairs):
        a = np.random.randint(low, high)
        b = np.random.randint(low, high)
        if (a, b) not in output:
            output.add((a, b))
            n_gend += 1
            
            yield (a, b)


class RFF(TransformerMixin):
    def __init__(self, output_dim, sigma=None):
        # output_dim - dimension of new feature space

        self.sigma = sigma
        self.output_dim = output_dim

        self.W = None
        self.b = None


    def __call__(self, X, y=None):
        return self.transform(X, y)


    def fit(self, X, y=None, n_pairs=1000):
        # n_pairs - number of sample pairs from population
        # to estimate parameter sigma with: sigma**2 = median(squared_norm(x_i-x_j)) for all i != j

        n_samples, input_dim = X.shape


        if self.sigma is None:
            sigma = self.estimate_sigma_(X, n_pairs)

        if self.W is None:
            # generate W from N(0, 1/sigma**2) distribution
            self.W = 1 / sigma * np.random.randn(input_dim, self.output_dim) + 0

        if self.b is None:
            self.b = 2*np.pi * np.random.uniform(self.output_dim) - np.pi

        return self


    def transform(self, X, y = None):
        if self.W is None or self.b is None:
            raise NotFittedError("One of parameters self.W or self.b is None. Probably you forgot to run fit() method firstly.")

        assert X.shape[1] == self.W.shape[0], "Last dimension of the input data: %s should match first dimension of weight matrix: %s" % (X.shape, self.W.shape)
        
        X_rff = np.cos(X@self.W + self.b)

        return X_rff


    def estimate_sigma_(self, X, n_pairs):
        squared_distances = np.zeros(n_pairs)
        
        idx = 0
        for i, j in gen_random_pair(0, len(X), n_pairs):
            squared_distances[idx] = ((X[i] - X[j])**2).sum()
            idx += 1

        sigma = np.sqrt(np.median(squared_distances))

        return sigma