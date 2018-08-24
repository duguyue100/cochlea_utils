from __future__ import division

from scipy.stats import norm
import warnings
import numpy as np
import progressbar
from matplotlib import pyplot

index_angles_01 = np.array([[12, 0], [11, -30], [9, -90], [8, -60], [4, 60], [3, 90], [1, 30]])
index_angles_02 = np.array([[1, 30], [2, 60], [3, 90], [4, 120], [5, 150], [6, 180], [7, 210], [8, 240], [9, 270],
                            [10, 300], [11, 330], [12, 0]])


def estimate(itds, initial_estimate, transition_probabilities, itd_dict, prior, save=None, verbose=False):
    localization_estimate = initial_estimate
    num_itds = len(itds)
    estimates = np.zeros(shape=(num_itds, prior.shape[0]), dtype=np.float32)
    argmax_estimates = np.zeros(shape=num_itds, dtype=np.int32)
    bar = progressbar.ProgressBar() if verbose else identity
    for itd_idx, itd in bar(enumerate(itds)):
        position_matrix = np.multiply(transition_probabilities, localization_estimate)
        position_probability = np.sum(position_matrix, axis=1)
        motion_probability = np.array([prior[idx][np.argmin(np.abs(itd_dict - itd))] for idx in range(prior.shape[0])])
        probability_to_normalize = np.multiply(motion_probability, position_probability)
        localization_estimate = probability_to_normalize / sum(probability_to_normalize)
        estimates[itd_idx] = localization_estimate
        argmax_estimates[itd_idx] = np.argmax(localization_estimate)
        if np.isnan(np.sum(localization_estimate)):
            warnings.warn('Something wrong with the estimate.')
    if save is not None:
        np.savez(save, estimates=estimates, argmax_estimates=argmax_estimates)
    return np.array(estimates, dtype=np.float32), np.array(argmax_estimates, dtype=np.float)


def get_priors(itd_streams, max_itd=800e-6, num_bins=80, save_to_file=None):
    priors = np.zeros(shape=(len(itd_streams), num_bins), dtype=np.float32)
    for idx, itd_stream in enumerate(itd_streams):
        hist = np.histogram(itd_stream, bins=num_bins, range=(-max_itd, max_itd))[0] / len(itd_stream)
        priors[idx] = hist
    if save_to_file is not None:
        np.save(save_to_file, priors)
    return priors


def get_transition_probabilities(index_angles=index_angles_01, sigma=5):
    transition_probabilities = np.zeros(shape=(len(index_angles), len(index_angles)), dtype=np.float32)
    angles_original = index_angles[:, 1]
    angles = np.sort(angles_original)
    for angle_index, index_angle in enumerate(index_angles):
        mean = index_angle[1]
        angle_distribution = norm(mean, sigma).pdf(angles)
        angle_dict = {}
        for idx, angle in enumerate(angles):
            angle_dict[angle] = angle_distribution[idx]
        angle_distribution = [angle_dict[angle] for angle in angles_original]
        transition_probabilities[angle_index] = angle_distribution
    return transition_probabilities


def moving_average(estimates, window_length=10):
    averaged_estimates = np.zeros_like(estimates)
    for idx in range(len(estimates) - window_length + 1):
        averaged_estimates[idx] = np.mean(estimates[idx:idx + window_length], axis=0)
    for idx in range(len(estimates) - window_length + 1, len(estimates)):
        averaged_estimates[idx] = averaged_estimates[len(estimates) - window_length]
    return averaged_estimates


def identity(x):
    return x


if __name__ == '__main__':
    test_index_angles = np.array([[12, 0], [11, -30], [9, -90], [8, -60], [4, 60], [3, 90], [1, 30]])
    test_transition_probabilities = get_transition_probabilities(test_index_angles, sigma=5)
    pyplot.imshow(test_transition_probabilities, aspect='auto', interpolation='nearest')
    pyplot.show()
    print('Hello world, nothing to test for now.')
