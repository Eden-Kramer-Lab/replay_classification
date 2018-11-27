from sklearn.base import BaseEstimator

import numpy as np
from numba import jit


@jit(nogil=True, nopython=True, cache=True)
def _normal_pdf(x, mean=0, std_deviation=1):
    '''Evaluate the normal probability density function at specified points.
    Unlike the `scipy.stats.norm.pdf`, this function is not general and does
    not do any sanity checking of the inputs. As a result it is a much faster
    function, but you should be sure of your inputs before using.
    This function only computes the one-dimensional pdf.
    Parameters
    ----------
    x : array_like
        The normal probability function will be evaluated
    mean : float or array_like, optional
    std_deviation : float or array_like
    Returns
    -------
    probability_density
        The normal probability density function evaluated at `x`
    '''
    return (np.exp(-0.5 * ((x - mean) / std_deviation) ** 2) /
            (np.sqrt(2.0 * np.pi) * std_deviation))


class IsotropicKernelDensity(BaseEstimator):
    def __init__(self, bandwidth):
        self.bandwidth = np.array(bandwidth)

    def fit(self, X, y=None):
        """Fit the Kernel Density model on the data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.
        """
        self.training_data = X

        return self

    def score_samples(self, X):
        """Evaluate the density model on the data.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            An array of points to query.  Last dimension should match dimension
            of training data (n_features).

        Returns
        -------
        density : ndarray, shape (n_samples,)
            The array of log(density) evaluations.
        """
        bandwidth = self.bandwidth[-X.shape[1]:]
        return np.log(np.mean(np.prod(_normal_pdf(
            X[:, np.newaxis, :], mean=self.training_data[np.newaxis, ...],
            std_deviation=bandwidth), axis=2), axis=1))

    def score(self, X, y=None):
        """Compute the total log probability under the model.

        Parameters
        ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points.  Each row
            corresponds to a single data point.

        Returns
        -------
        logprob : float
            Total log-likelihood of the data in X.
        """
        return np.sum(self.score_samples(X))

    def sample(self, n_samples=1, random_state=None):
        """Generate random samples from the model.

        Currently, this is implemented only for gaussian and tophat kernels.

        Parameters
        ----------
        n_samples : int, optional
            Number of samples to generate. Defaults to 1.
        random_state : int, RandomState instance or None. default to None
            If int, random_state is the seed used by the random number
            generator; If RandomState instance, random_state is the random
            number generator; If None, the random number generator is the
            RandomState instance used by `np.random`.

        Returns
        -------
        X : array_like, shape (n_samples, n_features)
            List of samples.
        """
        raise NotImplementedError()
