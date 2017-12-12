import numpy as np
from pytest import mark
from scipy.linalg import block_diag

from replay_classification.core import (_fix_zero_bins, _get_prior,
                                        _normalize_row_probability,
                                        _update_posterior,
                                        combined_likelihood,
                                        get_bin_centers,
                                        normalize_to_probability)


def test__normalize_row_probability():
    '''All rows should sum to one after normalization
    '''
    transition_matrix = np.arange(1, 10).reshape(3, 3)
    expected = np.ones((3,))
    assert np.allclose(
        _normalize_row_probability(transition_matrix).sum(axis=1),
        expected)


def test__fix_zero_bins():
    '''A column of all zeros should be set to ones
    '''
    transition_matrix = np.arange(0, 9).reshape(3, 3)
    transition_matrix[:, 2] = 0
    expected = np.copy(transition_matrix)
    expected[:, 2] = 1
    assert np.allclose(_fix_zero_bins(transition_matrix), expected)


def test_normalize_to_probability():
    ''' A vector should normalize to one
    '''
    x = np.arange(1, 9)
    assert normalize_to_probability(x).sum() == 1


@mark.parametrize('data, exponent, expected', [
    (np.arange(1, 9), 1, 1),
    (np.arange(1, 9), 2, 1),  # test kwarg
    (np.array([[0.2, 0.4], [0.1, 0.2]]), 2,
     np.array([np.exp(-0.15), 1])),
    (2, 2, 1),  # test single data point
])
def test_combined_likelihood(data, exponent, expected):
    def likelihood_function(x, exponent=1):
        return x ** exponent
    assert np.allclose(
        combined_likelihood(data, likelihood_function,
                            likelihood_kwargs=dict(exponent=exponent)),
        expected)


def test__update_posterior():
    prior1 = 2 * np.ones((2,))
    prior2 = np.ones((3,))
    prior = np.hstack((prior1, prior2))

    likelihood1 = 3 * np.ones((2,))
    likelihood2 = 3 * np.ones((3,))
    likelihood = np.hstack((likelihood1, likelihood2))

    posterior = _update_posterior(prior, likelihood)
    expected = np.ones((5,))
    expected[:2] = 6 / 21
    expected[2:] = 3 / 21

    assert np.allclose(posterior, expected)


def test__get_prior():
    posterior1 = 2 * np.ones((2,))
    posterior2 = np.ones((3,))
    posterior = np.hstack((posterior1, posterior2))

    state_transition1 = 3 * np.ones((2, 2))
    state_transition2 = 4 * np.ones((3, 3))
    state_transition = block_diag(
        state_transition1, state_transition2)
    prior = _get_prior(posterior, state_transition)
    expected = 12 * np.ones((5,))

    assert np.allclose(prior, expected)


@mark.parametrize('bin_edges, expected', [
    (np.arange(0, 5), np.arange(0, 4) + 0.5),
    (np.arange(0, 12, 2), np.arange(1, 10, 2))
]
)
def test_get_bin_centers(bin_edges, expected):
    bin_centers = get_bin_centers(bin_edges)
    assert np.allclose(bin_centers, expected)
