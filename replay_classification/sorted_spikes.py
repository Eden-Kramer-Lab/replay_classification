from logging import getLogger

import numpy as np
import pandas as pd
from patsy import build_design_matrices, dmatrix
from statsmodels.api import families

from regularized_glm import penalized_IRLS

try:
    from IPython import get_ipython

    if get_ipython() is not None:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)

logger = getLogger(__name__)


def fit_glm_model(spikes, design_matrix, penalty=3):
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
    regularization_weights = np.ones((design_matrix.shape[1],)) * penalty
    regularization_weights[0] = 0.0
    return np.squeeze(
        penalized_IRLS(
            np.array(design_matrix), np.array(spikes),
            family=families.Poisson(),
            penalty=regularization_weights).coefficients)


def predictors_by_experimental_condition(experimental_condition,
                                         place_bin_centers,
                                         design_matrix):
    '''The design matrix for a given trajectory direction.
    '''
    predictors = {'position': place_bin_centers,
                  'experimental_condition': [experimental_condition] *
                  len(place_bin_centers)}
    return build_design_matrices(
        [design_matrix.design_info], predictors)[0]


def get_conditional_intensity(fit_coefficients, predict_design_matrix):
    '''The conditional intensity for each model
    '''
    intensity = np.exp(np.dot(predict_design_matrix, fit_coefficients)).T
    intensity[np.isnan(intensity)] = np.spacing(1)
    return intensity


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
    conditional_intensity += np.spacing(1)
    probability_no_spike = -conditional_intensity * time_bin_size
    is_spike = atleast_kd(is_spike, conditional_intensity.ndim)
    return (np.log(conditional_intensity) * is_spike +
            probability_no_spike)


def fit_spike_observation_model(position, experimental_condition, spikes,
                                place_bin_centers,
                                knot_spacing, observation_state_order,
                                spike_model_penalty=1E-1):
    min_position, max_position = np.nanmin(position), np.nanmax(position)
    n_steps = (max_position - min_position) // knot_spacing
    position_knots = min_position + np.arange(1, n_steps) * knot_spacing
    formula = ('1 + experimental_condition * '
               'cr(position, knots=position_knots, constraints="center")')

    training_data = pd.DataFrame(dict(
        position=position,
        experimental_condition=experimental_condition))
    design_matrix = dmatrix(
        formula, training_data, return_type='dataframe')
    fit_coefficients = np.stack(
        [fit_glm_model(
            pd.DataFrame(s).loc[design_matrix.index], design_matrix,
            spike_model_penalty)
         for s in tqdm(spikes.T, desc='neurons')], axis=1)

    ci_by_state = {
        condition: get_conditional_intensity(
            fit_coefficients, predictors_by_experimental_condition(
                condition, place_bin_centers, design_matrix))
        for condition in np.unique(observation_state_order)}

    conditional_intensity = np.stack(
        [ci_by_state[state] for state in observation_state_order],
        axis=1)
    return conditional_intensity[:, np.newaxis, ...]
