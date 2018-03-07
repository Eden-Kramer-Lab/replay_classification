from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr

import holoviews as hv

from .clusterless import (fit_clusterless_observation_model,
                          poisson_mark_log_likelihood,
                          estimate_marginalized_joint_mark_intensity)
from .core import (combined_likelihood,
                   get_bin_centers, inbound_outbound_initial_conditions,
                   predict_state, uniform_initial_conditions)
from .sorted_spikes import fit_spike_observation_model, poisson_log_likelihood
from .state_transition import fit_state_transition

logger = getLogger(__name__)

_DEFAULT_STATE_NAMES = ['Outbound-Forward', 'Outbound-Reverse',
                        'Inbound-Forward', 'Inbound-Reverse']

_DEFAULT_OBSERVATION_STATE_ORDER = ['Outbound', 'Outbound',
                                    'Inbound', 'Inbound']

_DEFAULT_STATE_TRANSITION_STATE_ORDER = ['Outbound', 'Inbound',
                                         'Inbound', 'Outbound']
hv.extension('bokeh', 'matplotlib')


class ClusterlessDecoder(object):
    '''

    Attributes
    ----------
    position : ndarray, shape (n_time,)
        Position of the animal to train the model on.
    lagged_position : ndarray, shape (n_time,)
        Position of the animal at the previous time step
    trajectory_direction : array_like, shape (n_time,)
        Task of the animal. Element must be either
         'Inbound' or 'Outbound'.
    spike_marks : ndarray, shape (n_signals, n_time, n_marks)
        Marks to train the model on.
        If spike does not occur, the row must be marked with np.nan
    n_position_bins : int, optional
    mark_std_deviation : float, optional
    replay_speedup_factor : int, optional
    observation_state_order : list of str, optional
    state_transition_state_order : list of str, optional
    initial_conditions : 'Inbound-Outbound' | 'Uniform' | dict of array,
        optional
    time_bin_size : float, optional
    confidence_threshold : float, optional

    '''

    def __init__(self, position, lagged_position,
                 trajectory_direction, spike_marks,
                 n_position_bins=61, mark_std_deviation=20,
                 replay_speedup_factor=16,
                 state_names=_DEFAULT_STATE_NAMES,
                 observation_state_order=_DEFAULT_OBSERVATION_STATE_ORDER,
                 state_transition_state_order=_DEFAULT_STATE_TRANSITION_STATE_ORDER,
                 initial_conditions='Inbound-Outbound',
                 time_bin_size=1,
                 place_std_deviation=None,
                 confidence_threshold=0.8):
        self.position = np.array(position)
        self.lagged_position = np.array(lagged_position)
        self.trajectory_direction = np.array(trajectory_direction)
        self.spike_marks = np.array(spike_marks)
        self.n_position_bins = n_position_bins
        self.mark_std_deviation = mark_std_deviation
        self.replay_speedup_factor = replay_speedup_factor
        self.state_names = state_names
        self.observation_state_order = observation_state_order
        self.state_transition_state_order = state_transition_state_order
        self.initial_conditions = initial_conditions
        self.time_bin_size = time_bin_size
        self.place_std_deviation = place_std_deviation
        self.confidence_threshold = confidence_threshold

    def fit(self):
        '''Fits the decoder model for each trajectory_direction.

        Relates the position and spike_marks to the trajectory_direction.

        Parameters
        ----------


        Returns
        -------
        self : class instance

        '''

        self.place_bin_edges = np.linspace(
            self.position.min(), self.position.max(),
            self.n_position_bins + 1)
        if self.place_std_deviation is None:
            self.place_std_deviation = (
                (self.position.max() - self.position.min()) /
                self.n_position_bins)
        self.place_bin_centers = get_bin_centers(self.place_bin_edges)

        trajectory_directions = np.unique(
            self.trajectory_direction[pd.notnull(self.trajectory_direction)])

        if self.initial_conditions == 'Inbound-Outbound':
            self.initial_conditions = inbound_outbound_initial_conditions(
                self.place_bin_centers)
        elif self.initial_conditions == 'Uniform':
            self.initial_conditions = {
                direction: uniform_initial_conditions(self.place_bin_centers)
                for direction in trajectory_directions}

        self.initial_conditions = np.stack(
            [self.initial_conditions[state]
             for state in self.state_transition_state_order]
        ) / len(self.state_names)
        self.initial_conditions = xr.DataArray(
            self.initial_conditions, dims=['state', 'position'],
            coords=dict(position=self.place_bin_centers,
                        state=self.state_names),
            name='probability')

        logger.info('Fitting state transition model...')
        self.state_transition_matrix = fit_state_transition(
            self.position, self.lagged_position, self.place_bin_edges,
            self.place_bin_centers, self.trajectory_direction,
            trajectory_directions, self.replay_speedup_factor,
            self.state_transition_state_order, self.state_names)

        logger.info('Fitting observation model...')
        joint_mark_intensity_functions, ground_process_intensity = (
            fit_clusterless_observation_model(
                self.position, self.trajectory_direction, self.spike_marks,
                self.place_bin_centers, trajectory_directions,
                self.place_std_deviation, self.mark_std_deviation,
                self.observation_state_order))

        likelihood_kwargs = dict(
            joint_mark_intensity_functions=joint_mark_intensity_functions,
            ground_process_intensity=ground_process_intensity,
            time_bin_size=self.time_bin_size)

        self._combined_likelihood_kwargs = dict(
            log_likelihood_function=poisson_mark_log_likelihood,
            likelihood_kwargs=likelihood_kwargs)

        return self

    def plot_initial_conditions(self, **kwargs):
        return (
            self.initial_conditions.to_series().unstack().T.plot(**kwargs))

    def plot_state_transition_model(self, **kwargs):
        try:
            return (self.state_transition_matrix
                    .plot(x='position_t', y='position_t_1', col='state',
                          robust=True, **kwargs))
        except ValueError:
            return (self.state_transition_matrix
                        .plot(x='position_t', y='position_t_1',
                              robust=True, **kwargs))

    def marginalized_intensities(self):
        joint_mark_intensity_functions = (
            self._combined_likelihood_kwargs['likelihood_kwargs']
            ['joint_mark_intensity_functions'])
        mark_bin_centers = np.linspace(100, 350, 200)

        marginalized_intensities = np.stack(
            [[estimate_marginalized_joint_mark_intensity(
                mark_bin_centers, jmi.keywords['training_marks'],
                jmi.keywords['place_field'],
                jmi.keywords['place_occupancy'], self.mark_std_deviation)
              for jmi in tetrode]
             for tetrode in joint_mark_intensity_functions])
        dims = ['signal', 'state', 'position', 'marks', 'mark_dimension']
        coords = dict(
            state=self.state_names,
            marks=mark_bin_centers,
            position=self.place_bin_centers
        )
        return xr.DataArray(marginalized_intensities, dims=dims,
                            coords=coords)

    def plot_observation_model(self, sampling_frequency=1):
        marginalized_intensities = (
            self.marginalized_intensities().sum('mark_dimension')
            * sampling_frequency)
        try:
            return marginalized_intensities.plot(
                row='signal', col='state', x='position', y='marks',
                robust=True)
        except ValueError:
            return marginalized_intensities.plot(
                row='signal', x='position', y='marks', robust=True)

    def save_model():
        raise NotImplementedError

    def load_model():
        raise NotImplementedError

    def predict(self, spike_marks, time=None):
        '''Predicts the state from spike_marks.

        Parameters
        ----------
        spike_marks : ndarray, shape (n_signals, n_time, n_marks)
            If spike does not occur, the row must be marked with np.nan.
        time : ndarray, optional, shape (n_time,)

        Returns
        -------
        predicted_state : str

        '''
        results = predict_state(
            spike_marks,
            initial_conditions=self.initial_conditions.values,
            state_transition=self.state_transition_matrix.values,
            likelihood_function=combined_likelihood,
            likelihood_kwargs=self._combined_likelihood_kwargs)
        coords = dict(
            time=(time if time is not None
                  else np.arange(results['posterior_density'].shape[0])),
            position=self.place_bin_centers,
            state=self.state_names
        )

        DIMS = ['time', 'state', 'position']

        results = xr.Dataset(
            {key: (DIMS, value) for key, value in results.items()},
            coords=coords)

        return DecodingResults(
            results=results,
            spikes=spike_marks,
            confidence_threshold=self.confidence_threshold,
        )


class SortedSpikeDecoder(object):

    def __init__(self, position, lagged_position, spikes, trajectory_direction,
                 n_position_bins=61, replay_speedup_factor=16,
                 state_names=_DEFAULT_STATE_NAMES,
                 observation_state_order=_DEFAULT_OBSERVATION_STATE_ORDER,
                 state_transition_state_order=_DEFAULT_STATE_TRANSITION_STATE_ORDER,
                 initial_conditions='Inbound-Outbound',
                 time_bin_size=1,
                 knot_spacing=30,
                 confidence_threshold=0.8):
        '''

        Attributes
        ----------
        position : ndarray, shape (n_time,)
        lagged_position : ndarray, shape (n_time,)
        spike : ndarray, shape (n_neurons, n_time)
        trajectory_direction : ndarray, shape (n_time,)
        n_position_bins : int, optional
        replay_speedup_factor : int, optional
        observation_state_order : list of str, optional
        state_transition_state_order : list of str, optional
        initial_conditions : 'Inbound-Outbound' | 'Uniform' | dict of array,
            optional
        time_bin_size : float, optional
        confidence_threshold : float, optional

        '''
        self.position = position
        self.lagged_position = lagged_position
        self.trajectory_direction = trajectory_direction
        self.spikes = spikes
        self.n_position_bins = n_position_bins
        self.replay_speedup_factor = replay_speedup_factor
        self.state_names = state_names
        self.observation_state_order = observation_state_order
        self.state_transition_state_order = state_transition_state_order
        self.initial_conditions = initial_conditions
        self.time_bin_size = time_bin_size
        self.confidence_threshold = confidence_threshold
        self.knot_spacing = knot_spacing

    def fit(self):
        '''Fits the decoder model by state

        Relates the position and spikes to the state.
        '''
        self.place_bin_edges = np.linspace(
            np.floor(self.position.min()), np.ceil(self.position.max()),
            self.n_position_bins + 1)
        self.place_bin_centers = get_bin_centers(self.place_bin_edges)

        trajectory_directions = np.unique(
            self.trajectory_direction[pd.notnull(self.trajectory_direction)])

        if self.initial_conditions == 'Inbound-Outbound':
            self.initial_conditions = inbound_outbound_initial_conditions(
                self.place_bin_centers)
        elif self.initial_conditions == 'Uniform':
            self.initial_conditions = {
                direction: uniform_initial_conditions(self.place_bin_centers)
                for direction in trajectory_directions}

        self.initial_conditions = np.stack(
            [self.initial_conditions[state]
             for state in self.state_transition_state_order]
        ) / len(self.state_names)
        self.initial_conditions = xr.DataArray(
            self.initial_conditions, dims=['state', 'position'],
            coords=dict(position=self.place_bin_centers,
                        state=self.state_names),
            name='probability')

        logger.info('Fitting state transition model...')
        self.state_transition_matrix = fit_state_transition(
            self.position, self.lagged_position, self.place_bin_edges,
            self.place_bin_centers, self.trajectory_direction,
            trajectory_directions, self.replay_speedup_factor,
            self.state_transition_state_order, self.state_names)

        logger.info('Fitting observation model...')
        conditional_intensity = fit_spike_observation_model(
            self.position, self.trajectory_direction, self.spikes,
            self.place_bin_centers, trajectory_directions,
            self.knot_spacing, self.observation_state_order)

        self._combined_likelihood_kwargs = dict(
            log_likelihood_function=poisson_log_likelihood,
            likelihood_kwargs=dict(conditional_intensity=conditional_intensity)
        )

        return self

    def save_model():
        raise NotImplementedError

    def load_model():
        raise NotImplementedError

    def plot_initial_conditions(self, **kwargs):
        return (
            self.initial_conditions.to_series().unstack().T.plot(**kwargs))

    def plot_state_transition_model(self, **kwargs):
        try:
            return (self.state_transition_matrix
                    .plot(x='position_t', y='position_t_1', col='state',
                          robust=True, **kwargs))
        except ValueError:
            return (self.state_transition_matrix
                        .plot(x='position_t', y='position_t_1',
                              robust=True, **kwargs))

    def plot_observation_model(self, sampling_frequency=1):
        conditional_intensity = self._combined_likelihood_kwargs[
            'likelihood_kwargs']['conditional_intensity']
        coords = dict(
            state=self.state_names,
            position=self.place_bin_centers)
        conditional_intensity = xr.DataArray(
            conditional_intensity * sampling_frequency,
            coords=coords,
            dims=['signal', 'state', 'position'],
            name='firing_rate').to_dataframe().reset_index()
        g = sns.FacetGrid(
            conditional_intensity,
            row='signal', col='state')
        return g.map(plt.plot, 'position', 'firing_rate')

    def predict(self, spikes, time=None):
        '''Predicts the state from spike_marks.

        Parameters
        ----------
        spikes : ndarray, shape (n_neurons, n_time)
            If spike does not occur, the row must be marked with np.nan.
        time : ndarray, optional, shape (n_time,)

        Returns
        -------
        predicted_state : str

        '''
        results = predict_state(
            spikes,
            initial_conditions=self.initial_conditions.values,
            state_transition=self.state_transition_matrix.values,
            likelihood_function=combined_likelihood,
            likelihood_kwargs=self._combined_likelihood_kwargs)
        coords = dict(
            time=(time if time is not None
                  else np.arange(results['posterior_density'].shape[0])),
            position=self.place_bin_centers,
            state=self.state_names
        )
        DIMS = ['time', 'state', 'position']

        results = xr.Dataset(
            {key: (DIMS, value) for key, value in results.items()},
            coords=coords)

        return DecodingResults(
            results=results,
            spikes=spikes,
            confidence_threshold=self.confidence_threshold,
        )


class DecodingResults():

    def __init__(self, results, spikes=None, confidence_threshold=0.8):
        self.results = results
        self.spikes = spikes
        self.confidence_threshold = confidence_threshold

    def state_probability(self):
        return (self.results['posterior_density'].sum('position')
                .to_series().unstack())

    def predicted_state(self):
        state_probability = self.state_probability()
        is_threshold = np.sum(
            (state_probability > self.confidence_threshold), axis=1)
        if np.any(is_threshold):
            return state_probability.loc[is_threshold.argmax()].argmax()
        else:
            return 'Unclassified'

    def predicted_state_probability(self):
        state_probability = self.state_probability()
        is_threshold = np.sum(
            (state_probability > self.confidence_threshold), axis=1)
        if np.any(is_threshold):
            return state_probability.loc[is_threshold.argmax()].max()
        else:
            return np.nan

    def plot_posterior_density(self, **kwargs):
        try:
            return self.results['posterior_density'].plot(
                x='time', y='position', col='state', col_wrap=2,
                robust=True, **kwargs)
        except ValueError:
            return self.results['posterior_density'].plot(
                x='time', y='position', robust=True, **kwargs)

    def plot_interactive(self):
        ds = hv.Dataset(self.results)
        likelihood_plot = ds.to(hv.Curve, 'position', 'likelihood')
        posterior_plot = ds.to(hv.Curve, 'position', 'posterior_density')
        prior_plot = ds.to(hv.Curve, 'position', 'prior')
        plot_opts = dict(shared_yaxis=True, shared_xaxis=True)
        norm_opts = dict(framewise=True)
        hv.opts({'Curve': dict(norm=norm_opts)}, likelihood_plot)
        return (prior_plot.grid('state').opts(plot=plot_opts) +
                likelihood_plot.grid('state').opts(plot=plot_opts) +
                posterior_plot.grid('state').opts(plot=plot_opts)).cols(1)

    def plot_state_probability(self, **kwargs):
        return self.state_probability().plot(**kwargs)
