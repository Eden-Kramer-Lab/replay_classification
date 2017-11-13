from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from patsy import dmatrix

import holoviews as hv

from .clusterless import (build_joint_mark_intensity,
                          estimate_ground_process_intensity,
                          estimate_marginalized_joint_mark_intensity,
                          poisson_mark_likelihood)
from .core import (combined_likelihood, empirical_movement_transition_matrix,
                   get_bin_centers, inbound_outbound_initial_conditions,
                   predict_state, uniform_initial_conditions)
from .sorted_spikes import (get_conditional_intensity, glm_fit,
                            poisson_likelihood,
                            predictors_by_trajectory_direction)

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

    '''

    def __init__(self, position, trajectory_direction, spike_marks,
                 n_position_bins=61, mark_std_deviation=20,
                 replay_speedup_factor=16,
                 state_names=_DEFAULT_STATE_NAMES,
                 observation_state_order=_DEFAULT_OBSERVATION_STATE_ORDER,
                 state_transition_state_order=_DEFAULT_STATE_TRANSITION_STATE_ORDER,
                 initial_conditions='Inbound-Outbound',
                 time_bin_size=1,
                 place_std_deviation=None):
        self.position = np.array(position)
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
            self.place_std_deviation = 2 * np.diff(self.place_bin_edges)[0]
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

        state_transition_by_state = {
            direction: empirical_movement_transition_matrix(
                self.position,
                self.place_bin_edges, self.replay_speedup_factor,
                np.in1d(self.trajectory_direction, direction))
            for direction in trajectory_directions}
        state_transition_matrix = np.stack(
            [state_transition_by_state[state]
             for state in self.state_transition_state_order])
        self.state_transition_matrix = xr.DataArray(
            state_transition_matrix,
            dims=['state', 'position_t', 'position_t_1'],
            coords=dict(state=self.state_names,
                        position_t=self.place_bin_centers,
                        position_t_1=self.place_bin_centers),
            name='state_transition_probability')

        logger.info('Fitting observation model...')
        joint_mark_intensity_functions = []
        ground_process_intensity = []

        for marks in self.spike_marks:
            jmi_by_state = {
                direction: build_joint_mark_intensity(
                    self.position[
                        np.in1d(self.trajectory_direction, direction)],
                    marks[np.in1d(self.trajectory_direction, direction)],
                    self.place_bin_centers, self.place_std_deviation,
                    self.mark_std_deviation)
                for direction in trajectory_directions}
            joint_mark_intensity_functions.append(
                [jmi_by_state[state]
                 for state in self.observation_state_order])

            gpi_by_state = {
                direction: estimate_ground_process_intensity(
                    self.position[
                        np.in1d(self.trajectory_direction, direction)],
                    marks[np.in1d(self.trajectory_direction, direction)],
                    self.place_bin_centers, self.place_std_deviation)
                for direction in trajectory_directions}
            ground_process_intensity.append(
                [gpi_by_state[state]
                 for state in self.observation_state_order])

        ground_process_intensity = np.stack(ground_process_intensity)
        likelihood_kwargs = dict(
            joint_mark_intensity_functions=joint_mark_intensity_functions,
            ground_process_intensity=ground_process_intensity,
            time_bin_size=self.time_bin_size)

        self._combined_likelihood_kwargs = dict(
            likelihood_function=poisson_mark_likelihood,
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

    def plot_observation_model(self):
        marginalized_intensities = (
            self.marginalized_intensities().sum('mark_dimension'))
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
        )


class SortedSpikeDecoder(object):

    def __init__(self, position, spikes, trajectory_direction,
                 n_position_bins=61, replay_speedup_factor=16,
                 state_names=_DEFAULT_STATE_NAMES,
                 observation_state_order=_DEFAULT_OBSERVATION_STATE_ORDER,
                 state_transition_state_order=_DEFAULT_STATE_TRANSITION_STATE_ORDER,
                 initial_conditions='Inbound-Outbound',
                 time_bin_size=1):
        '''

        Attributes
        ----------
        position : ndarray, shape (n_time,)
        spike : ndarray, shape (n_neurons, n_time)
        trajectory_direction : ndarray, shape (n_time,)
        n_position_bins : int, optional
        replay_speedup_factor : int, optional
        observation_state_order : list of str, optional
        state_transition_state_order : list of str, optional
        initial_conditions : 'Inbound-Outbound' | 'Uniform' | dict of array,
            optional
        time_bin_size : float, optional

        '''
        self.position = position
        self.trajectory_direction = trajectory_direction
        self.spikes = spikes
        self.n_position_bins = n_position_bins
        self.replay_speedup_factor = replay_speedup_factor
        self.state_names = state_names
        self.observation_state_order = observation_state_order
        self.state_transition_state_order = state_transition_state_order
        self.initial_conditions = initial_conditions
        self.time_bin_size = time_bin_size

    def fit(self):
        '''Fits the decoder model by state

        Relates the position and spikes to the state.
        '''
        self.place_bin_edges = np.linspace(
            np.floor(self.position.min()), np.ceil(self.position.max()),
            self.n_position_bins + 1)
        self.place_bin_centers = get_bin_centers(self.place_bin_edges)

        trajectory_directions = np.unique(self.trajectory_direction)

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

        state_transition_by_state = {
            direction: empirical_movement_transition_matrix(
                self.position,
                self.place_bin_edges, self.replay_speedup_factor,
                np.in1d(self.trajectory_direction, direction))
            for direction in trajectory_directions}
        state_transition_matrix = np.stack(
            [state_transition_by_state[state]
             for state in self.state_transition_state_order])
        self.state_transition_matrix = xr.DataArray(
            state_transition_matrix,
            dims=['state', 'position_t', 'position_t_1'],
            coords=dict(state=self.state_names,
                        position_t=self.place_bin_centers,
                        position_t_1=self.place_bin_centers),
            name='state_transition_probability')

        logger.info('Fitting observation model...')
        formula = ('1 + trajectory_direction * '
                   'bs(position, df=5, degree=3)')

        training_data = pd.DataFrame(dict(
            position=self.position,
            trajectory_direction=self.trajectory_direction))
        design_matrix = dmatrix(
            formula, training_data, return_type='dataframe')
        fit = [glm_fit(spikes, design_matrix, ind)
               for ind, spikes in enumerate(self.spikes)]

        ci_by_state = {
            direction: get_conditional_intensity(
                fit, predictors_by_trajectory_direction(
                    direction, self.place_bin_centers, design_matrix))
            for direction in trajectory_directions}
        conditional_intensity = np.stack(
            [ci_by_state[state] for state in self.observation_state_order],
            axis=1)
        self._combined_likelihood_kwargs = dict(
            likelihood_function=poisson_likelihood,
            likelihood_kwargs=dict(
                conditional_intensity=conditional_intensity)
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

    def plot_observation_model(self):
        conditional_intensity = self._combined_likelihood_kwargs[
            'likelihood_kwargs']['conditional_intensity']
        coords = dict(
            state=self.state_names,
            position=self.place_bin_centers)
        conditional_intensity = xr.DataArray(
            conditional_intensity,
            coords=coords,
            dims=['signal', 'state', 'position'],
            name='firing_rate').to_dataframe().reset_index()
        g = sns.FacetGrid(
            conditional_intensity, row='signal', col='state')
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
            spikes=spikes
        )


class DecodingResults():

    def __init__(self, results, spikes=None):
        self.results = results
        self.spikes = spikes

    def state_probability(self):
        return self.results['posterior_density'].sum('position').to_series().unstack()

    def predicted_state(self):
        return self.state_probability().iloc[-1].argmax()

    def predicted_state_probability(self):
        return self.state_probability().iloc[-1].max()

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
