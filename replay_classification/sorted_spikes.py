from logging import getLogger

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from statsmodels.api import families
from regularized_glm import penalized_IRLS
from patsy import dmatrix

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


def get_conditional_intensity(fit_coefficients, predict_design_matrix):
    '''The conditional intensity for each model
    '''
    return np.exp(np.dot(predict_design_matrix, fit_coefficients)).T


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


def fit_spike_observation_model(position, trajectory_direction, spikes,
                                place_bin_centers, trajectory_directions,
                                knot_spacing, observation_state_order):
    min_position, max_position = (position.min(), position.max())
    n_steps = (max_position - min_position) // knot_spacing
    position_knots = min_position + (np.arange(1, n_steps)
                                     * knot_spacing)
    formula = ('1 + trajectory_direction * '
               'cr(position, knots=position_knots, constraints="center")')

    training_data = pd.DataFrame(dict(
        position=position,
        trajectory_direction=trajectory_direction))
    design_matrix = dmatrix(
        formula, training_data, return_type='dataframe')
    fit_coefficients = np.stack(
        [fit_glm_model(
            pd.DataFrame(s).loc[design_matrix.index], design_matrix)
         for s in spikes], axis=1)

    ci_by_state = {
        direction: get_conditional_intensity(
            fit_coefficients, predictors_by_trajectory_direction(
                direction, place_bin_centers, design_matrix))
        for direction in trajectory_directions}
    return np.stack(
        [ci_by_state[state] for state in observation_state_order],
        axis=1)
