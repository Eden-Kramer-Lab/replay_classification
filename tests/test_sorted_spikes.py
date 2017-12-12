import numpy as np
from pytest import mark

from replay_classification.sorted_spikes import poisson_log_likelihood

CONDITIONAL_INTENSITY = np.atleast_2d(np.array([0.20, 0.50, 0.20, 0.25]))
SPIKE_LOG_LIKELIHOOD = np.log(CONDITIONAL_INTENSITY) - CONDITIONAL_INTENSITY


@mark.parametrize('is_spike, expected_likelihood', [
    (np.zeros(3,), -CONDITIONAL_INTENSITY),
    (np.array([0, 1, 0]), np.concatenate(
        (-CONDITIONAL_INTENSITY,
         SPIKE_LOG_LIKELIHOOD,
         -CONDITIONAL_INTENSITY))),
    (np.ones(3,), SPIKE_LOG_LIKELIHOOD * np.ones((3, 1))),
])
def test_poisson_likelihood_is_spike(is_spike, expected_likelihood):
    ci = CONDITIONAL_INTENSITY * np.ones((3, 1))
    likelihood = poisson_log_likelihood(
        is_spike, conditional_intensity=ci,
        time_bin_size=1)
    assert np.allclose(likelihood, expected_likelihood)
