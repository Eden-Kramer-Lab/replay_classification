'''Classifying sharp-wave ripple replay events from spiking activity
(e.g. Forward vs. Reverse replay)

References
----------
.. [1] Deng, X., Liu, D.F., Karlsson, M.P., Frank, L.M., and Eden, U.T.
       (2016). Rapid classification of hippocampal replay content for
       real-time applications. Journal of Neurophysiology 116, 2221-2235.

'''

from functools import partial

import numpy as np
import pandas as pd

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


def poisson_mark_log_likelihood(marks, joint_mark_intensity_functions=None,
                                ground_process_intensity=None,
                                time_bin_size=1):
    '''Probability of parameters given spiking indicator at a particular
    time and associated marks.

    Parameters
    ----------
    marks : array, shape (n_signals, n_marks)
    joint_mark_intensity : function
        Instantaneous probability of observing a spike given mark vector
        from data. The parameters for this function should already be set,
        before it is passed to `poisson_mark_log_likelihood`.
    ground_process_intensity : array, shape (n_signals, n_states,
                                             n_place_bins)
        Probability of observing a spike regardless of marks.
    time_bin_size : float, optional

    Returns
    -------
    poisson_mark_log_likelihood : array_like, shape (n_signals, n_states,
                                                     n_time, n_place_bins)

    '''
    ground_process_intensity += np.spacing(1)
    probability_no_spike = -ground_process_intensity * time_bin_size
    joint_mark_intensity = np.stack(
        [np.stack([jmi(signal_marks) for jmi in jmi_by_state], axis=-2)
         for signal_marks, jmi_by_state
         in zip(tqdm(marks, 'electrodes'), joint_mark_intensity_functions)])
    joint_mark_intensity += np.spacing(1)
    return np.log(joint_mark_intensity) + probability_no_spike


def joint_mark_intensity(
        marks, place_bin_centers, place_occupancy, fitted_model, mean_rate):
    marks = np.atleast_2d(marks)
    n_place_bins = place_bin_centers.shape[0]
    n_time = marks.shape[0]
    is_nan = np.any(np.isnan(marks), axis=1)
    n_spikes = np.sum(~is_nan)
    density = np.zeros((n_time, n_place_bins))

    if n_spikes > 0:
        for bin_ind, bin in enumerate(place_bin_centers):
            bin = bin * np.ones((n_spikes, 1))
            predict_data = np.concatenate((marks[~is_nan], bin), axis=1)
            density[~is_nan, bin_ind] = np.exp(
                fitted_model.score_samples(predict_data))

    joint_mark_intensity = mean_rate * density / place_occupancy
    joint_mark_intensity[is_nan] = 1.0
    return joint_mark_intensity


def estimate_place_occupancy(position, place_bin_centers, model, model_kwargs):
    return np.exp(model(**model_kwargs).fit(position)
                  .score_samples(place_bin_centers[:, np.newaxis]))


def estimate_ground_process_intensity(
        position, marks, place_bin_centers, model, model_kwargs):
    is_spike = np.any(~np.isnan(marks), axis=1)
    position = atleast_2d(position)
    place_field = np.exp(model(**model_kwargs).fit(position[is_spike])
                         .score_samples(place_bin_centers[:, np.newaxis]))
    place_occupancy = estimate_place_occupancy(
        position, place_bin_centers, model, model_kwargs)
    mean_rate = np.mean(is_spike)
    return np.atleast_2d(mean_rate * place_field / place_occupancy)


def build_joint_mark_intensity(
        position, training_marks, place_bin_centers, model, model_kwargs):
    training_marks = atleast_2d(training_marks)[~np.isnan(position)]
    position = atleast_2d(position)[~np.isnan(position)]

    is_spike = np.any(~np.isnan(training_marks), axis=1)
    mean_rate = np.mean(is_spike, dtype=np.float)

    training_data = np.concatenate(
        (training_marks[is_spike], position[is_spike]), axis=1)
    fitted_model = model(**model_kwargs).fit(training_data)
    place_occupancy = estimate_place_occupancy(
        position, place_bin_centers, model, model_kwargs)

    return partial(joint_mark_intensity,
                   place_bin_centers=place_bin_centers,
                   place_occupancy=place_occupancy,
                   fitted_model=fitted_model,
                   mean_rate=mean_rate)


def fit_multiunit_observation_model(position, trajectory_direction,
                                    spike_marks, place_bin_centers,
                                    model, model_kwargs,
                                    observation_state_order):
    joint_mark_intensity_functions = []
    ground_process_intensity = []

    trajectory_directions = np.unique(
        trajectory_direction[pd.notnull(trajectory_direction)])

    for marks in tqdm(spike_marks, desc='electrodes'):
        jmi_by_state = {
            direction: build_joint_mark_intensity(
                position[
                    np.in1d(trajectory_direction, direction)],
                marks[np.in1d(trajectory_direction, direction)],
                place_bin_centers, model, model_kwargs)
            for direction in trajectory_directions}
        joint_mark_intensity_functions.append(
            [jmi_by_state[state]
             for state in observation_state_order])

        gpi_by_state = {
            direction: estimate_ground_process_intensity(
                position[
                    np.in1d(trajectory_direction, direction)],
                marks[np.in1d(trajectory_direction, direction)],
                place_bin_centers, model, model_kwargs)
            for direction in trajectory_directions}
        ground_process_intensity.append(
            np.stack([gpi_by_state[state]
                      for state in observation_state_order], axis=-2))

    ground_process_intensity = np.stack(ground_process_intensity)

    return joint_mark_intensity_functions, ground_process_intensity


def atleast_2d(x):
    return np.atleast_2d(x).T if x.ndim < 2 else x
