import numpy as np
from pytest import mark
from scipy.stats import multivariate_normal, norm

from replay_classification.clusterless import (_normal_pdf,
                                               estimate_place_field,
                                               estimate_place_occupancy,
                                               evaluate_mark_space)


def test_evaluate_mark_space():
    '''Tests that the mark space estimator puts a multivariate Gaussian
    at each mark.
    '''
    n_marks, n_training_spikes, mark_std_deviation = 4, 10, 1

    test_marks = np.arange(1, 9, 2)

    training_marks = np.zeros((n_training_spikes, n_marks))
    training_marks[3, :] = np.arange(9, 17, 2)

    mark_space_estimator = evaluate_mark_space(
        test_marks, training_marks=training_marks,
        mark_std_deviation=mark_std_deviation)

    expected_mark1 = multivariate_normal(
        mean=np.arange(9, 17, 2),
        cov=np.identity(n_marks) * mark_std_deviation).pdf(
            np.arange(1, 9, 2))
    expected_mark2 = multivariate_normal(
        mean=np.zeros(n_marks,),
        cov=np.identity(n_marks) * mark_std_deviation).pdf(
            np.arange(1, 9, 2))
    expected = np.ones((n_training_spikes,)) * expected_mark2
    expected[3] = expected_mark1

    assert np.allclose(mark_space_estimator, expected)


def test_estimate_place_field():
    '''Tests that there is a Gaussian centered around each given place
    at spike
    '''
    place_bins = np.linspace(0, 150, 61)
    is_spike = np.array([1, 1, 0], dtype=bool)
    position = np.asarray([25, 100, 30])
    place_std_deviation = 20
    place_field_estimator = estimate_place_field(
        position, is_spike, place_bins,
        place_std_deviation=place_std_deviation)

    expected1 = norm.pdf(
        place_bins, position[0], place_std_deviation)
    expected2 = norm.pdf(
        place_bins, position[1], place_std_deviation)

    assert np.allclose(place_field_estimator,
                       np.stack((expected1, expected2)).T)


def test_estimate_place_occupancy():
    '''Tests that there is a Gaussian centered around each given place
    '''
    place_bins = np.linspace(0, 150, 61)
    place = np.asarray([25, 100])
    place_std_deviation = 20
    place_occupancy = estimate_place_occupancy(
        place, place_bins, place_std_deviation)
    expected1 = norm.pdf(
        place_bins, place[0], place_std_deviation)
    expected2 = norm.pdf(
        place_bins, place[1], place_std_deviation)
    assert np.allclose(place_occupancy, expected1 + expected2)


@mark.parametrize('x, mean, std_deviation', [
    (np.asarray([-1, 1, 100]), 0, 1),
    (np.asarray([-1, 1, 100]), 100, 25),
    (np.asarray([-1, 1, 100]), np.asarray([0, 100, 10]),
     np.asarray([2, 5, 3])),
])
def test__normal_pdf(x, mean, std_deviation):
    expected = norm.pdf(x, loc=mean, scale=std_deviation)
    assert np.allclose(
        _normal_pdf(x, mean=mean, std_deviation=std_deviation),
        expected)
