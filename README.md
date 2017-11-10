# replay_classification
[![Coverage Status](https://coveralls.io/repos/github/Eden-Kramer-Lab/replay_classification/badge.svg?branch=master)](https://coveralls.io/github/Eden-Kramer-Lab/replay_classification?branch=master) [![DOI](https://zenodo.org/badge/104356770.svg)](https://zenodo.org/badge/latestdoi/104356770)

`replay_classification` is a python package for categorizing hippocampal replay events using multiunit spiking activity. Multiunit spiking activity can be more informative than sorted spikes, because there is no need to distinguish between neurons. This has several advantages:
1. We can take advantage of partial information from neurons that are not well separated in terms of electrophysiological signal.
2. Saves time, because no spike-sorting step is necessary.

This package also provides:
+  Metrics for the confidence of classification, allowing for experimental intervention before a replay event is completed.

![Probability of States](/state_probability.png)
+  Convenient functions for diagnostic plotting.

![Posterior Density](/replay_example.png)

See the notebooks ([\#1](https://nbviewer.jupyter.org/github/Eden-Kramer-Lab/replay_classification/blob/master/examples/Simulate_Ripple_Decoding_Data_Sorted_Spikes.ipynb), [\#2](https://nbviewer.jupyter.org/github/Eden-Kramer-Lab/replay_classification/blob/master/examples/Simulate_Ripple_Decoding_Data_Clusterless.ipynb) for more information on how to use the package.

### References and Citation ###
Please cite:
> Deng, X., Liu, D.F., Karlsson, M.P., Frank, L.M., and Eden, U.T.
(2016). Rapid classification of hippocampal replay content for
real-time applications. Journal of Neurophysiology 116, 2221-2235.

and the DOI for the package repository (see the DOI badge at the start of the README).

### Package Requirements ###
- python>=3.5
- numpy
- scipy
- pandas
- xarray
- statsmodels
- matplotlib
- seaborn
- patsy
- numba
- holoviews

### Installation ###

```python
pip install replay_classification
```
Or
```python
conda install -c edeno replay_classification
```

### Developer Installation ###

1. Install miniconda (or anaconda) if it isn't already installed. Type into bash (or install from the anaconda website):
```bash
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
```

2. Go to the local repository (`.../replay_classification`) and install the anaconda environment for the repository. Type into bash:
```bash
conda update -q conda
conda info -a
conda env create -f environment.yml
source activate replay_classification
python setup.py develop
```

### Example Usage ###

1. Classify replays into Inbound-Forward, Inbound-Reverse, Outbound-Forward, Outboud-Reverse using the clusterless decoder:
```python
from replay_classification import ClusterlessDecoder

decoder = ClusterlessDecoder(
    position=linear_distance,
    trajectory_direction=trajectory_direction,
    spike_marks=test_marks,
    replay_speedup_factor=16,
)

decoder.fit()

results = decoder.predict(ripple_marks)
```
2. Classify replays using sorted spikes:
```python
from replay_classification import SortedSpikeDecoder

decoder = SortedSpikeDecoder(
    position=linear_distance,
    trajectory_direction=trajectory_direction,
    spikes=test_spikes,
    replay_speedup_factor=16,
)

decoder.fit()

results = decoder.predict(ripple_spikes)
```
3. Decode only Inbound and Outbound by specifying the observation and state transition order:
```python
from replay_classification import ClusterlessDecoder

decoder = ClusterlessDecoder(
    position=linear_distance,
    trajectory_direction=trajectory_direction,
    spike_marks=test_marks,
    observation_state_order=['Inbound', 'Outbound'],
    state_transition_state_order=['Inbound', 'Outbound'],
    state_names=['Inbound', 'Outbound'],
    initial_conditions='Uniform',
)

decoder.fit()

results = decoder.predict(ripple_spikes)
```
4. Decode replay position:
```python
from replay_classification import ClusterlessDecoder

decoder = ClusterlessDecoder(
    position=linear_distance,
    trajectory_direction=np.ones_like(position), spike_marks=test_marks,
    observation_state_order=[1],
    state_transition_state_order=[1],
    state_names=['replay_position'],
    initial_conditions='Uniform',
)

decoder.fit()

results = decoder.predict(ripple_spikes)
```
