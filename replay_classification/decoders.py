from logging import getLogger

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from sklearn.neighbors import KernelDensity

import holoviews as hv

from .core import (combined_likelihood, filter, get_bin_centers, get_bin_edges,
                   inbound_outbound_initial_conditions, smooth,
                   uniform_initial_conditions)
from .multiunit import (fit_multiunit_observation_model,
                        poisson_mark_log_likelihood)
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


class _DecoderBase(BaseEstimator):
    def __init__(
        self, n_place_bins=None, place_bin_size=1,
        replay_speedup_factor=20, state_names=_DEFAULT_STATE_NAMES,
        observation_state_order=_DEFAULT_OBSERVATION_STATE_ORDER,
        state_transition_state_order=_DEFAULT_STATE_TRANSITION_STATE_ORDER,
            time_bin_size=1, confidence_threshold=0.8):
        '''

        Attributes
        ----------
        n_place_bins : None or int, optional
        place_bin_size : None or int, optional
        replay_speedup_factor : int, optional
        state_names : list of str, optional
        observation_state_order : list of str, optional
        state_transition_state_order : list of str, optional
        time_bin_size : float, optional
        confidence_threshold : float, optional

        '''
        self.n_place_bins = n_place_bins
        self.place_bin_size = place_bin_size
        self.replay_speedup_factor = replay_speedup_factor
        self.state_names = state_names
        self.observation_state_order = observation_state_order
        self.state_transition_state_order = state_transition_state_order
        self.time_bin_size = time_bin_size
        self.confidence_threshold = confidence_threshold

    def __dir__(self):
        return self.keys()

    def fit_place_bins(self, position, place_bin_edges=None):
        if place_bin_edges is not None:
            self.place_bin_edges = place_bin_edges
        else:
            self.place_bin_edges = get_bin_edges(
                position, self.n_place_bins, self.place_bin_size)
        self.place_bin_centers = get_bin_centers(self.place_bin_edges)

    def fit_initial_conditions(self, trajectory_direction,
                               initial_conditions='Inbound-Outbound'):
        '''

        Parameters
        ----------
        trajectory_direction : ndarray, shape (n_time,)
        initial_conditions : str or dict of ndarray, optional
        '''
        trajectory_directions = np.unique(
            trajectory_direction[pd.notnull(trajectory_direction)])

        if isinstance(initial_conditions, str):
            if initial_conditions == 'Inbound-Outbound':
                initial_conditions = inbound_outbound_initial_conditions(
                    self.place_bin_centers)
            elif initial_conditions == 'Uniform':
                initial_conditions = {
                    direction: uniform_initial_conditions(
                        self.place_bin_centers)
                    for direction in trajectory_directions}

        self.initial_conditions_ = np.stack(
            [initial_conditions[state]
             for state in self.state_transition_state_order]
        ) / len(self.state_names)
        self.initial_conditions_ = xr.DataArray(
            self.initial_conditions_, dims=['state', 'position'],
            coords=dict(position=self.place_bin_centers,
                        state=self.state_names),
            name='probability')

    def fit_state_transition(self, position, is_training, trajectory_direction,
                             replay_speedup_factor=20):
        logger.info('Fitting state transition model...')
        self.replay_speedup_factor = replay_speedup_factor
        df = pd.DataFrame({'position': position,
                           'is_training': is_training,
                           'trajectory_direction': trajectory_direction})
        df['lagged_position'] = df.position.shift(1)
        df['future_position'] = df.position.shift(-1)
        df = df.loc[df.is_training].dropna()

        self.state_transition_ = fit_state_transition(
            df['position'], df['lagged_position'], self.place_bin_edges,
            self.place_bin_centers, df['trajectory_direction'],
            self.replay_speedup_factor,
            self.state_transition_state_order, self.state_names)

        self.backwards_state_transition_ = fit_state_transition(
            df['future_position'], df['position'], self.place_bin_edges,
            self.place_bin_centers, df['trajectory_direction'],
            self.replay_speedup_factor,
            self.state_transition_state_order, self.state_names)

    def save_model(self, filename='model.pkl'):
        joblib.dump(self, filename)

    @staticmethod
    def load_model(filename='model.pkl'):
        return joblib.load(filename)

    def plot_initial_conditions(self, **kwargs):
        return (
            self.initial_conditions_.to_series().unstack().T.plot(**kwargs))

    def plot_state_transition_model(self, **kwargs):
        try:
            return (self.state_transition_
                    .plot(x='position_t', y='position_t_1', col='state',
                          robust=True, **kwargs))
        except ValueError:
            return (self.state_transition_
                        .plot(x='position_t', y='position_t_1',
                              robust=True, **kwargs))

    def plot_observation_model(self, sampling_frequency=1):
        raise NotImplementedError


class ClusterlessDecoder(_DecoderBase):
    '''

    Attributes
    ----------
    n_place_bins : None or int, optional
    place_bin_size : None or int, optional
    replay_speedup_factor : int, optional
    state_names : list of str, optional
    observation_state_order : list of str, optional
    state_transition_state_order : list of str, optional
    time_bin_size : float, optional
    confidence_threshold : float, optional

    '''

    def __init__(
        self, n_place_bins=None, place_bin_size=1,
        replay_speedup_factor=20, state_names=_DEFAULT_STATE_NAMES,
        observation_state_order=_DEFAULT_OBSERVATION_STATE_ORDER,
        state_transition_state_order=_DEFAULT_STATE_TRANSITION_STATE_ORDER,
        time_bin_size=1, confidence_threshold=0.8,
            model=KernelDensity, model_kwargs=dict(bandwidth=10)):
        super().__init__(n_place_bins, place_bin_size,
                         replay_speedup_factor, state_names,
                         observation_state_order,
                         state_transition_state_order,
                         time_bin_size, confidence_threshold)
        self.model = model
        self.model_kwargs = model_kwargs

    def fit(self, position, trajectory_direction,
            multiunits, is_training=None,
            initial_conditions='Inbound-Outbound', place_bin_edges=None):
        '''Fits the decoder model for each trajectory_direction.

        Relates the position and multiunits to the trajectory_direction.

        Parameters
        ----------
        position : ndarray, shape (n_time,)
        trajectory_direction : ndarray, shape (n_time,)
        multiunits : ndarray, shape (n_signals, n_time, n_features)
        is_training : None or ndarray, bool, shape (n_time, )
        initial_conditions : str or dict of ndarray, optional
        place_bin_edges : None or ndarray, optional

        Returns
        -------
        self : class instance

        '''

        self.fit_place_bins(position, place_bin_edges)
        self.fit_initial_conditions(trajectory_direction, initial_conditions)
        if is_training is None:
            is_training = np.ones_like(position, dtype=np.bool)
        self.fit_state_transition(
            position, is_training, trajectory_direction,
            self.replay_speedup_factor)

        logger.info('Fitting observation model...')
        joint_mark_intensity_functions, ground_process_intensity = (
            fit_multiunit_observation_model(
                position[is_training], trajectory_direction[is_training],
                multiunits[:, is_training], self.place_bin_centers,
                self.model, self.model_kwargs, self.observation_state_order))

        likelihood_kwargs = dict(
            joint_mark_intensity_functions=joint_mark_intensity_functions,
            ground_process_intensity=ground_process_intensity,
            time_bin_size=self.time_bin_size)

        self.combined_likelihood_kwargs_ = dict(
            log_likelihood_function=poisson_mark_log_likelihood,
            likelihood_kwargs=likelihood_kwargs)

        return self

    def predict(self, multiunits, time=None, is_smooth=True):
        '''Predicts the state from multiunits.

        Parameters
        ----------
        multiunits : ndarray, shape (n_signals, n_time, n_marks)
            If spike does not occur, the row must be marked with np.nan.
        time : ndarray, optional, shape (n_time,)
        is_smooth : bool
            Use future information to compute state

        Returns
        -------
        DecodingResults : DecodingResults class instance

        '''
        likelihood = combined_likelihood(
            multiunits, **self.combined_likelihood_kwargs_)
        place_bin_size = np.diff(self.place_bin_edges)[0]
        state_transition = self.state_transition_.values
        results = filter(
            initial_conditions=self.initial_conditions_.values,
            state_transition=state_transition,
            likelihood=likelihood,
            bin_size=place_bin_size)

        if is_smooth:
            results = smooth(
                filter_posterior=results['posterior_density'],
                backwards_state_transition=(self.backwards_state_transition_
                                            .values),
                bin_size=place_bin_size)
            results['likelihood'] = likelihood

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
            spikes=multiunits,
            place_bin_size=place_bin_size,
            confidence_threshold=self.confidence_threshold,
        )


class SortedSpikeDecoder(_DecoderBase):

    def __init__(
        self, n_place_bins=None, place_bin_size=1,
        replay_speedup_factor=20, state_names=_DEFAULT_STATE_NAMES,
        observation_state_order=_DEFAULT_OBSERVATION_STATE_ORDER,
        state_transition_state_order=_DEFAULT_STATE_TRANSITION_STATE_ORDER,
        time_bin_size=1, confidence_threshold=0.8, knot_spacing=15,
            spike_model_penalty=1E-1):
        '''

        Attributes
        ----------
        n_place_bins : None or int, optional
        place_bin_size : None or int, optional
        replay_speedup_factor : int, optional
        state_names : list of str, optional
        observation_state_order : list of str, optional
        state_transition_state_order : list of str, optional
        time_bin_size : float, optional
        confidence_threshold : float, optional
        knot_spacing : float, optional
        spike_model_penalty : float, optional

        '''
        super().__init__(n_place_bins, place_bin_size,
                         replay_speedup_factor, state_names,
                         observation_state_order,
                         state_transition_state_order,
                         time_bin_size, confidence_threshold)
        self.knot_spacing = knot_spacing
        self.spike_model_penalty = spike_model_penalty

    def fit(self, position, trajectory_direction, spikes, is_training=None,
            initial_conditions='Inbound-Outbound', place_bin_edges=None):
        '''Fits the decoder model by state

        Relates the position and spikes to the state.

        Parameters
        ----------
        position : ndarray, shape (n_time,)
        trajectory_direction : ndarray, shape (n_time,)
        spikes : ndarray, shape (n_time, n_neurons)
        is_training : None or ndarray, bool, shape (n_time, )
        initial_conditions : str or dict of ndarray, optional
        place_bin_edges : None or ndarray, optional

        Returns
        -------
        self : class instance

        '''
        self.fit_place_bins(position, place_bin_edges)
        self.fit_initial_conditions(trajectory_direction, initial_conditions)
        if is_training is None:
            is_training = np.ones_like(position, dtype=np.bool)
        self.fit_state_transition(
            position, is_training, trajectory_direction,
            self.replay_speedup_factor)

        logger.info('Fitting observation model...')
        trajectory_directions = np.unique(
            trajectory_direction[pd.notnull(trajectory_direction)])
        conditional_intensity = fit_spike_observation_model(
            position[is_training], trajectory_direction[is_training],
            spikes[is_training], self.place_bin_centers, trajectory_directions,
            self.knot_spacing, self.observation_state_order,
            self.spike_model_penalty)

        self.combined_likelihood_kwargs_ = dict(
            log_likelihood_function=poisson_log_likelihood,
            likelihood_kwargs=dict(conditional_intensity=conditional_intensity)
        )

        return self

    def plot_observation_model(self, sampling_frequency=1):
        '''

        Parmameters
        -----------
        sampling_frequency : float, optional

        '''
        conditional_intensity = self.combined_likelihood_kwargs_[
            'likelihood_kwargs']['conditional_intensity'].squeeze()
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

    def predict(self, spikes, time=None, is_smooth=True):
        '''Predicts the state from multiunits.

        Parameters
        ----------
        spikes : ndarray, shape (n_time, n_neurons)
            If spike does not occur, the row must be marked with np.nan.
        time : ndarray, optional, shape (n_time,)
        is_smooth : bool
            Use future information to compute state

        Returns
        -------
        DecodingResults : DecodingResults class instance

        '''
        logger.info('Predicting replay type...')
        place_bin_size = np.diff(self.place_bin_edges)[0]
        likelihood = combined_likelihood(
            spikes.T[..., np.newaxis, np.newaxis],
            **self.combined_likelihood_kwargs_)
        state_transition = self.state_transition_.values
        results = filter(
            initial_conditions=self.initial_conditions_.values,
            state_transition=state_transition,
            likelihood=likelihood,
            bin_size=place_bin_size)
        if is_smooth:
            results = smooth(
                filter_posterior=results['posterior_density'],
                backwards_state_transition=(self.backwards_state_transition_
                                            .values),
                bin_size=place_bin_size)
            results['likelihood'] = likelihood
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
            place_bin_size=place_bin_size,
            confidence_threshold=self.confidence_threshold,
        )


class DecodingResults():

    def __init__(self, results, place_bin_size, spikes=None,
                 confidence_threshold=0.8):
        self.results = results
        self.spikes = spikes
        self.place_bin_size = place_bin_size
        self.confidence_threshold = confidence_threshold

    def __dir__(self):
        return self.keys()

    def state_probability(self):
        return (self.results['posterior_density'].sum('position')
                .to_series().unstack()) * self.place_bin_size

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
