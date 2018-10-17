import numpy as np
import xarray as xr
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import norm

from .core import normalize_to_probability


def uniform_initial_conditions(
    position_info, place_bin_edges, place_bin_centers,
        replay_sequence_orders='Forward'):
    if isinstance(replay_sequence_orders, str):
        replay_sequence_orders = [replay_sequence_orders]

    bin_size = place_bin_centers[1] - place_bin_centers[0]
    uniform = normalize_to_probability(
        np.ones_like(place_bin_centers), bin_size)

    state_names = []
    initial_conditions = []

    for order in replay_sequence_orders:
        for condition, df in position_info.groupby('experimental_condition'):
            state_names.append('-'.join((condition, order)))
            initial_conditions.append(uniform)

    initial_conditions = normalize_to_probability(
        np.stack(initial_conditions), bin_size)

    return xr.DataArray(
        initial_conditions, dims=['state', 'position'],
        coords=dict(position=place_bin_centers,
                    state=state_names),
        name='probability')


def inbound_outbound_initial_conditions(place_bin_centers):
    '''Sets the prior for each state (Outbound-Forward, Outbound-Reverse,
    Inbound-Forward, Inbound-Reverse).

    Inbound states have greater weight on starting at the center arm.
    Outbound states have weight everywhere else.

    Parameters
    ----------
    place_bin_centers : ndarray, shape (n_bins,)
        Histogram bin centers of the place measure

    Returns
    -------
    initial_conditions : dict

    '''
    bin_size = place_bin_centers[1] - place_bin_centers[0]

    outbound_initial_conditions = normalize_to_probability(
        norm.pdf(place_bin_centers, loc=0,
                 scale=bin_size * 2), bin_size)

    inbound_initial_conditions = normalize_to_probability(
        (np.max(outbound_initial_conditions) *
         np.ones(place_bin_centers.shape)) -
        outbound_initial_conditions, bin_size)

    uniform = uniform_initial_conditions(place_bin_centers)

    return {'Inbound': inbound_initial_conditions,
            'Outbound': outbound_initial_conditions,
            'Stay': uniform}


def fit_initial_conditions(position_info, place_bin_edges, place_bin_centers,
                           replay_sequence_orders='Forward'):
    if isinstance(replay_sequence_orders, str):
        replay_sequence_orders = [replay_sequence_orders]
    order_to_position = {'Forward': lambda s: s.iloc[0],
                         'Reverse': lambda s: s.iloc[-1],
                         'Stay': lambda s: None}
    order_to_position = {order_name: func
                         for order_name, func in order_to_position.items()
                         if order_name in replay_sequence_orders}
    grouper = (position_info
               .groupby(['experimental_condition', 'trial_id'])
               .position.agg(order_to_position)
               .stack(dropna=False)
               .reset_index()
               .rename(columns={'level_2': 'order', 0: 'position'})
               .groupby(['order', 'experimental_condition']))

    state_names = []
    initial_conditions = []
    bin_size = np.diff(place_bin_edges)[0]

    for (order, condition), df in grouper:
        state_names.append('-'.join((condition, order)))
        initial_conditions.append(empirical_inital_conditions(
            df.position, place_bin_edges, bin_size))

    initial_conditions = normalize_to_probability(
        np.stack(initial_conditions), bin_size)

    return xr.DataArray(
        initial_conditions, dims=['state', 'position'],
        coords=dict(position=place_bin_centers,
                    state=state_names),
        name='probability')


def empirical_inital_conditions(position, place_bin_edges, sigma=0.5):
    try:
        movement_bins, _ = np.histogram(position, bins=place_bin_edges)
    except ValueError:
        movement_bins = np.zeros((place_bin_edges.size - 1, ))
    if movement_bins.sum() == 0:
        movement_bins[:] = 1

    bin_size = np.diff(place_bin_edges)[0]
    movement_bins = normalize_to_probability(movement_bins, bin_size)

    return gaussian_filter(movement_bins, sigma=sigma)
