'''Classifying sharp-wave ripple replay events from spiking activity
(e.g. Forward vs. Reverse replay)

References
----------
.. [1] Deng, X., Liu, D.F., Karlsson, M.P., Frank, L.M., and Eden, U.T.
       (2016). Rapid classification of hippocampal replay content for
       real-time applications. Journal of Neurophysiology 116, 2221-2235.

'''
from functools import wraps
from logging import getLogger

import numpy as np

logger = getLogger(__name__)


def filter(initial_conditions, state_transition, likelihood, bin_size):
    '''Adaptive filter to iteratively calculate the posterior probability
    of a state variable using past information.

    Parameters
    ----------
    initial_conditions : ndarray, shape (n_states, n_bins)
    state_transition : ndarray, shape (n_states, n_bins, n_bins)
    likelihood : ndarray, shape (n_time, n_states, n_bins)
    bin_size : float

    Returns
    -------
    results : dict

    '''
    likelihood = likelihood[..., np.newaxis]
    n_time = likelihood.shape[0]
    shape = (n_time, *initial_conditions.shape, 1)
    posterior = np.zeros(shape)
    prior = np.zeros(shape)

    posterior[0] = initial_conditions.copy()[..., np.newaxis]

    for time_ind in np.arange(1, n_time):
        prior[time_ind] = predict_state(
            posterior[time_ind - 1], state_transition, bin_size)
        posterior[time_ind] = update_posterior(
            prior[time_ind], likelihood[time_ind], bin_size)

    return {'posterior_density': posterior.squeeze(),
            'likelihood': likelihood.squeeze(),
            'prior': prior.squeeze()}


def smooth(filter_posterior, state_transition, bin_size):
    '''Uses past and future information to estimate the state.

    Parameters
    ----------
    filter_posterior : ndarray, shape (n_time, n_bins)
    state_transition : ndarray, shape (n_states, n_bins, n_bins)
    bin_size : float

    Return
    ------
    results : dict
    '''
    filter_posterior = filter_posterior[..., np.newaxis]
    smoother_posterior = np.zeros_like(filter_posterior)
    smoother_posterior[-1] = filter_posterior[-1].copy()
    smoother_prior = np.zeros_like(filter_posterior)
    n_time = filter_posterior.shape[0]

    for time_ind in np.arange(n_time - 2, -1, -1):
        smoother_prior[time_ind] = predict_state(
            filter_posterior[time_ind], state_transition,
            bin_size)
        smoother_posterior[time_ind] = update_backwards_posterior(
            filter_posterior[time_ind], state_transition,
            smoother_posterior[time_ind + 1], smoother_prior[time_ind],
            bin_size)

    return {'filter_posterior': filter_posterior.squeeze(),
            'posterior_density': smoother_posterior.squeeze(),
            'prior': smoother_prior.squeeze()}


def update_backwards_posterior(filter_posterior, state_transition,
                               smoother_posterior, prior, bin_size):
    '''

    Parameters
    ----------
    filter_posterior : ndarray, shape (n_states, n_bins, 1)
    state_transition : ndarray, shape (n_states, n_bins, n_bins)
    smoother_posterior : ndarray, shape (n_states, n_bins, 1)
    prior : ndarray, shape (n_states, n_bins, 1)
    bin_size : float

    Returns
    -------
    updated_posterior : ndarray, shape (n_states, n_bins)

    '''
    log_ratio = (np.log(smoother_posterior + np.spacing(1)) -
                 np.log(prior + np.spacing(1))).swapaxes(1, 2)
    weights = np.exp(log_ratio) @ state_transition * bin_size
    weights = weights.squeeze()[..., np.newaxis]
    return normalize_to_probability(weights * filter_posterior, bin_size)


def update_posterior(prior, likelihood, bin_size):
    '''The posterior density given the prior state weighted by the
    observed instantaneous likelihoodself.

    Parameters
    ----------
    prior : ndarray, shape (n_states, n_bins, 1)
    likelihood : ndarray, shape (n_states, n_bins, 1)

    Returns
    -------
    updated_posterior : ndarray, shape (n_states, n_bins, 1)

    '''
    return normalize_to_probability(prior * likelihood, bin_size)


def normalize_to_probability(distribution, bin_size):
    '''Ensure the distribution integrates to 1 so that it is a probability
    distribution
    '''
    return distribution / np.nansum(distribution) / bin_size


def predict_state(posterior, state_transition, bin_size):
    '''The prior given the previous posterior density and a transition
    matrix indicating the state at the next time step.
    '''
    return state_transition @ posterior * bin_size


def scaled_likelihood(log_likelihood_func):
    '''Converts a log likelihood to a scaled likelihood with its max value at
    1.

    Used primarily to keep the likelihood numerically stable because more
    observations at a time point will lead to a smaller overall likelihood
    and this can exceed the floating point accuarcy of a machine.

    Parameters
    ----------
    log_likelihood_func : function

    Returns
    -------
    scaled_likelihood : function

    '''
    @wraps(log_likelihood_func)
    def decorated_function(*args, **kwargs):
        log_likelihood = log_likelihood_func(*args, **kwargs)
        axis = tuple(range(log_likelihood.ndim)[1:])
        return np.exp(log_likelihood - np.max(
            log_likelihood, axis=axis, keepdims=True))

    return decorated_function


@scaled_likelihood
def combined_likelihood(data, log_likelihood_function=None,
                        likelihood_kwargs={}):
    '''Applies likelihood function to each signal and returns their product

    If there isn't a column dimension, just returns the likelihood.

    Parameters
    ----------
    data : ndarray, shape (n_signals, ...)
    log_likelihood_function : function
        Log Likelihood function to be applied to each signal.
        The likelihood function must take data as its first argument.
        All other arguments for the likelihood should be passed
        via `likelihood_kwargs`
    likelihood_kwargs : dict
        Keyword arguments for the likelihood function

    Returns
    -------
    likelihood : ndarray, shape (n_time, n_states, n_bins)

    '''
    try:
        return np.sum(
            log_likelihood_function(data, **likelihood_kwargs), axis=0)
    except ValueError:
        return log_likelihood_function(data, **likelihood_kwargs)


def get_bin_edges(position, n_bins=None, place_bin_size=None):
    not_nan_position = position[~np.isnan(position)]
    if place_bin_size is not None:
        n_bins = (np.round(np.ceil(np.ptp(not_nan_position) / place_bin_size))
                  ).astype(np.int)
    return np.linspace(
        np.min(not_nan_position), np.max(not_nan_position), n_bins + 1,
        endpoint=True)


def get_bin_centers(bin_edges):
    '''Given the outer-points of bins, find their center
    '''
    return bin_edges[:-1] + np.diff(bin_edges) / 2
