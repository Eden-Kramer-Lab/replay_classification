from logging import getLogger

import numpy as np
from patsy import build_design_matrices
from statsmodels.api import GLM, families

logger = getLogger(__name__)


def fit_glm_model(spikes, design_matrix, penalty=1E-5):
    '''Fits the Poisson model to the spikes from a neuron.

    Parameters
    ----------
    spikes : array_like
    design_matrix : array_like or pandas DataFrame
    ind : int
    penalty : float, optional

    Returns
    -------
    fitted_model : statsmodel results

    '''
    model = GLM(spikes, design_matrix, family=families.Poisson(),
                drop='missing')
    if penalty is None:
        return model.fit()
    else:
        regularization_weights = np.ones((design_matrix.shape[1],)) * penalty
        regularization_weights[0] = 0.0
        return model.fit_regularized(alpha=regularization_weights, L1_wt=0)


def predictors_by_trajectory_direction(trajectory_direction,
                                       place_bin_centers,
                                       design_matrix):
    '''The design matrix for a given trajectory direction.
    '''
    predictors = {'position': place_bin_centers,
                  'trajectory_direction': [trajectory_direction] *
                  len(place_bin_centers)}
    return build_design_matrices(
        [design_matrix.design_info], predictors)[0]


def glm_val(fitted_model, predict_design_matrix):
    '''Predict the model's response given a design matrix and the model
    parameters.

    '''
    try:
        return fitted_model.predict(predict_design_matrix)
    except AttributeError:
        return np.full(predict_design_matrix.shape[0], np.nan)


def get_conditional_intensity(fit, predict_design_matrix):
    '''The conditional intensity for each model
    '''
    return [glm_val(fitted_model, predict_design_matrix)
            for fitted_model in fit]


def atleast_kd(array, k):
    '''
    https://stackoverflow.com/questions/42516569/numpy-add-variable-number-of-dimensions-to-an-array
    '''
    new_shape = array.shape + (1,) * (k - array.ndim)
    return array.reshape(new_shape)


def poisson_log_likelihood(is_spike, conditional_intensity=None,
                           time_bin_size=1):
    '''Probability of parameters given spiking at a particular time

    Parameters
    ----------
    is_spike : array_like with values in {0, 1}, shape (n_signals,)
        Indicator of spike or no spike at current time.
    conditional_intensity : array_like, shape (n_signals, n_states,
                                               n_place_bins)
        Instantaneous probability of observing a spike
    time_bin_size : float, optional

    Returns
    -------
    poisson_log_likelihood : array_like, shape (n_signals, n_states,
                                                n_place_bins)

    '''
    probability_no_spike = -conditional_intensity * time_bin_size
    is_spike = atleast_kd(is_spike, conditional_intensity.ndim)
    conditional_intensity[
        np.isclose(conditional_intensity, 0.0)] = np.spacing(1)
    return (np.log(conditional_intensity) * is_spike +
            probability_no_spike)
