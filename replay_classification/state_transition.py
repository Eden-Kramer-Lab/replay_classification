import numpy as np
import xarray as xr
from .core import atleast_kd
from scipy.ndimage.filters import gaussian_filter
from statsmodels.api import GLM, families


def estimate_movement_std(position):
    '''Estimates the movement standard deviation based on position.

    WARNING: Need to use on original position, not interpolated position.

    Parameters
    ----------
    position : ndarray, shape (n_time, n_position_dim)

    Returns
    -------
    movement_std : ndarray, shape (n_position_dim,)

    '''
    position = atleast_kd(position, 2)
    is_nan = np.any(np.isnan(position), axis=1)
    position = position[~is_nan]
    movement_std = []
    for p in position.T:
        fit = GLM(p[:-1], p[1:], family=families.Gaussian()).fit()
        movement_std.append(np.sqrt(fit.scale))
    return np.array(movement_std)


def empirical_movement_transition_matrix(place, lagged_place, place_bin_edges,
                                         replay_speed=20,
                                         movement_std=0.5):
    '''Estimate the probablity of the next position based on the movement
     data, given the movment is sped up by the
     `replay_speed`

    Place cell firing during a hippocampal replay event is a "sped-up"
    version of place cell firing when the animal is actually moving.
    Here we use the animal's actual movements to constrain which place
    cell is likely to fire next.

    Parameters
    ----------
    place : array_like, shape (n_time,)
        Linearized position of the animal over time
    lagged_place : array_like, shape (n_time,)
        Linearized position of the preivous time step
    place_bin_edges : array_like, shape (n_bins,)
    replay_speed : int, optional
        How much the movement is sped-up during a replay event

    Returns
    -------
    empirical_movement_transition_matrix : array_like,
                                           shape=(n_bin_edges-1,
                                           n_bin_edges-1)

    '''
    movement_bins, _, _ = np.histogram2d(
        place, lagged_place,
        bins=(place_bin_edges, place_bin_edges))

    movement_bins = _fix_zero_bins(movement_bins)
    movement_bins = _normalize_row_probability(movement_bins)
    movement_bins = gaussian_filter(movement_bins, sigma=movement_std)

    return np.linalg.matrix_power(movement_bins, replay_speed)


def _normalize_row_probability(x):
    '''Ensure the state transition matrix rows sum to 1
    '''
    return x / x.sum(axis=1, keepdims=True)


def _fix_zero_bins(movement_bins):
    '''If there is no data observed for a column, set everything to 1 so
    that it will have equal probability
    '''
    movement_bins[:, movement_bins.sum(axis=0) == 0] = 1
    return movement_bins


def fit_state_transition(position_info, place_bin_edges, place_bin_centers,
                         replay_sequence_orders='Forward', replay_speed=20,
                         movement_std=0.5):
    order_to_df_column = {'Forward': 'lagged_position',
                          'Reverse': 'future_position',
                          'Stay': 'position'}
    if isinstance(replay_sequence_orders, str):
        replay_sequence_orders = [replay_sequence_orders]

    state_transition = []
    state_names = []

    for order in replay_sequence_orders:
        column_name = order_to_df_column[order]
        for condition, df in position_info.groupby('experimental_condition'):
            state_names.append('-'.join((condition, order)))
            state_transition.append(
                empirical_movement_transition_matrix(
                    df.position, df[column_name], place_bin_edges,
                    replay_speed=replay_speed,
                    movement_std=movement_std))

    state_transition = np.stack(state_transition)

    return xr.DataArray(
        state_transition,
        dims=['state', 'position_t', 'position_t_1'],
        coords=dict(
            state=state_names,
            position_t=place_bin_centers,
            position_t_1=place_bin_centers),
        name='state_transition_probability')
