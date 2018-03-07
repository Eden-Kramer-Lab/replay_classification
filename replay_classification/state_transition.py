import numpy as np
import xarray as xr
from scipy.ndimage.filters import gaussian_filter


def empirical_movement_transition_matrix(place, lagged_place, place_bin_edges,
                                         sequence_compression_factor=16,
                                         is_condition=None):
    '''Estimate the probablity of the next position based on the movement
     data, given the movment is sped up by the
     `sequence_compression_factor`

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
    sequence_compression_factor : int, optional
        How much the movement is sped-up during a replay event
    is_condition : array_like, shape (n_time,)
        Boolean indicator for an experimental condition.
    Returns
    -------
    empirical_movement_transition_matrix : array_like,
                                           shape=(n_bin_edges-1,
                                           n_bin_edges-1)

    '''
    if is_condition is None:
        is_condition = np.ones_like(place, dtype=bool)

    movement_bins, _, _ = np.histogram2d(
        place[is_condition], lagged_place[is_condition],
        bins=(place_bin_edges, place_bin_edges),
        normed=False)

    smoothed_movement_bins_probability = gaussian_filter(
        _normalize_row_probability(
            _fix_zero_bins(movement_bins)), sigma=0.5)
    return np.linalg.matrix_power(
        smoothed_movement_bins_probability,
        sequence_compression_factor)


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


def fit_state_transition(position, lagged_position, place_bin_edges,
                         place_bin_centers, trajectory_direction,
                         trajectory_directions, replay_speedup_factor,
                         state_transition_state_order, state_names):
    state_transition_by_state = {
        direction: empirical_movement_transition_matrix(
            position, lagged_position,
            place_bin_edges, replay_speedup_factor,
            np.in1d(trajectory_direction, direction))
        for direction in trajectory_directions}
    state_transition_matrix = np.stack(
        [state_transition_by_state[state]
         for state in state_transition_state_order])
    return xr.DataArray(
        state_transition_matrix,
        dims=['state', 'position_t', 'position_t_1'],
        coords=dict(state=state_names,
                    position_t=place_bin_centers,
                    position_t_1=place_bin_centers),
        name='state_transition_probability')
