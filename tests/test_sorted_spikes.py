import numpy as np
from pytest import mark

from replay_classification.sorted_spikes import poisson_likelihood


@mark.parametrize('is_spike, expected_likelihood', [
    (np.zeros(3,), np.array([[5, 2, 5, 4],
                             [5, 2, 5, 4],
                             [5, 2, 5, 4],
                             ])),
    (np.array([0, 1, 0]), np.array([[5, 2, 5, 4],
                                    [5 * np.log(0.2), 2 * np.log(0.5),
                                     5 * np.log(0.2), 4 * np.log(0.25)],
                                    [5, 2, 5, 4],
                                    ])),
    (np.ones(3,), np.array([[5 * np.log(0.2), 2 * np.log(0.5),
                             5 * np.log(0.2), 4 * np.log(0.25)],
                            [5 * np.log(0.2), 2 * np.log(0.5),
                             5 * np.log(0.2), 4 * np.log(0.25)],
                            [5 * np.log(0.2), 2 * np.log(0.5),
                             5 * np.log(0.2), 4 * np.log(0.25)],
                            ])),
])
def test_poisson_likelihood_is_spike(is_spike, expected_likelihood):
    conditional_intensity = np.array(
        [[np.log(0.2), np.log(0.5), np.log(0.2), np.log(0.25)],
         [np.log(0.2), np.log(0.5), np.log(0.2), np.log(0.25)],
         [np.log(0.2), np.log(0.5), np.log(0.2), np.log(0.25)]
         ])
    likelihood = poisson_likelihood(
        is_spike, conditional_intensity=conditional_intensity,
        time_bin_size=1)
    assert np.allclose(likelihood, expected_likelihood)
