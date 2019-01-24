from __future__ import print_function
from __future__ import absolute_import

"""
    Utility pre-processing feature generation for the spike data
"""

import numpy as np


def spike_count_features_by_time(timestamps, addresses, twl=0.005, tws=0.005, nb_channels=64, **options):
    """Creates spike count features for an audio event stream, by time binning.
    Args:
        :param timestamps: The timestamps in the event stream, a 1 dimensional numpy array of length nb_events.
        :param addresses: The addresses in the event stream, a 2 dimensional numpy array of addresses; the first column
        holds the channel address, while the second column holds the type of neuron.
        :param twl: The length of time window for bunching the events.
        :param tws: The shift of time window in bunching the events.
        :param nb_channels: The number of frequency channels.
    Returns:
        The spike count features for the event_stream; of shape (nb_time_bins, nb_channels).
    """
    time_in_stream = timestamps[-1] - timestamps[0]
    nb_bins = int(np.ceil(time_in_stream / tws))
    data_array_to_return = np.zeros(shape=(nb_bins, nb_channels), dtype='float32')
    current_time = timestamps[0]
    frame = 0
    while True:
        indices_left = current_time <= timestamps
        indices_right = timestamps < current_time + twl
        indices = np.multiply(indices_left, indices_right)
        if np.amax(indices):
            current_addresses = addresses[indices]
            data_array_to_return[frame] = np.histogram(current_addresses, bins=range(nb_channels + 1))[0] \
                .astype(np.float32, copy=False)
        frame += 1
        current_time += tws
        if frame == nb_bins:
            break
    return data_array_to_return


def exponential_features_by_time(timestamps, addresses, twl=0.005, tws=0.005, nb_channels=64, bunching='average',
                                 tau_type='constant', **options):
    """Creates time bunched exponential features for an audio event stream.
    The function first creates the exponential features for the events in the event stream through the function
    exponential_features. Then the events in every time bin are bunched together and the time bunched
    feature is created through either averaging or summing the features together.
    Args:
        :param timestamps: The timestamps in the event stream, a 1 dimensional numpy array of length nb_events.
        :param addresses: The addresses in the event stream, a 2 dimensional numpy array of addresses; the first column
        holds the channel address, while the second column holds the type of neuron.
        :param twl: The length of time window for bunching the events.
        :param tws: The shift of time window in bunching the events.
        :param nb_channels: The number of frequency channels.
        :param bunching: The mode of bunching the events. If 'average', the features for the events in each time bin are
        averaged, while if 'sum', the features for the events in each time bin are summed.
        :param tau_type: The type of tau to be used for the features.
    Returns:
        The time bunched exponential features for the event stream; of shape (nb_time_bins, nb_channels).
    """
    time_in_stream = np.amax(timestamps) - np.amin(timestamps)
    nb_bins = int(np.ceil(time_in_stream / tws))
    features_to_return = np.zeros(shape=(nb_bins, nb_channels), dtype='float32')
    if tau_type == 'constant':
        exp_data = exponential_features(timestamps, addresses, nb_channels=nb_channels, **options)
    else:
        exp_data = exponential_features(timestamps, addresses, nb_channels=nb_channels, **options)
    current_time = timestamps[0]
    frame = 0
    while True:
        indices_left = current_time <= timestamps
        indices_right = timestamps < current_time + twl
        indices = np.multiply(indices_left, indices_right)
        if np.amax(indices):
            current_events = exp_data[indices]
            if bunching == 'average':
                features_to_return[frame] = np.mean(current_events, axis=0)
            elif bunching == 'sum':
                features_to_return[frame] = np.sum(current_events, axis=0)
        else:
            features_to_return[frame] = np.zeros(shape=nb_channels, dtype='float32')
        frame += 1
        current_time += tws
        if frame == nb_bins:
            break
    return features_to_return


def spike_count_features_by_events(timestamps, addresses, ewl=100, ews=100, nb_channels=64, **options):
    """Creates spike count features for an audio event stream, by event binning.
    Args:
        :param timestamps: The timestamps in the event stream, a 1 dimensional numpy array of length nb_events.
        :param addresses: The addresses in the event stream, a 2 dimensional numpy array of addresses; the first column
        holds the channel address, while the second column holds the type of neuron.
        :param ewl: The number of events to bunch in a single frame.
        :param ews: The shift in number of events for bunching the events.
        :param nb_channels: The number of frequency channels.
    Returns:
        The event bunched spike count features for the event stream; of shape (nb_bins, nb_channels).
    """
    nb_bins = int(np.ceil(timestamps.shape[0] / ews))
    data_array_to_return = np.zeros(shape=(nb_bins, nb_channels), dtype='float32')
    frame = 0
    while frame < nb_bins:
        current_addresses = addresses[frame * ews:frame * ews + ewl]
        data_array_to_return[frame] = np.histogram(current_addresses, bins=range(nb_channels + 1))[0] \
            .astype(np.float32, copy=False)
        frame += 1
    return data_array_to_return


def exponential_features_by_events(timestamps, addresses, ewl=100, ews=100, nb_channels=64, bunching='average',
                                         tau_type='constant', **options):
    """Creates exponential features for an audio event stream, by event binning.
    Args:
        :param timestamps: The timestamps in the event stream, a 1 dimensional numpy array of length nb_events.
        :param addresses: The addresses in the event stream, a 2 dimensional numpy array of addresses; the first column
        holds the channel address, while the second column holds the type of neuron.
        :param ewl: The number of events to bunch in a single frame.
        :param ews: The shift in number of events for bunching the events.
        :param nb_channels: The number of frequency channels.
        :param bunching: The mode of bunching the events. If 'average', the features for the events in each time bin are
        averaged, while if 'sum', the features for the events in each time bin are summed.
        :param tau_type: The type of tau to be used for the features.
    Returns:
        The event bunched exponential features for the event stream; of shape (nb_bins, nb_channels).
    """
    nb_bins = int(np.ceil(timestamps.shape[0] / ews))
    data_array_to_return = np.zeros(shape=(nb_bins, nb_channels), dtype='float32')
    if tau_type == 'constant':
        exp_data = exponential_features(timestamps, addresses, nb_channels=nb_channels, **options)
    else:
        exp_data = exponential_features(timestamps, addresses, nb_channels=nb_channels, **options)
    frame = 0
    while frame < nb_bins:
        current_addresses = np.arange(frame * ews, frame * ews + ewl)
        current_events = exp_data[current_addresses]
        if bunching == 'average':
            data_array_to_return[frame] = np.mean(current_events, axis=0)
        elif bunching == 'sum':
            data_array_to_return[frame] = np.sum(current_events, axis=0)
        else:
            data_array_to_return[frame] = np.zeros(shape=nb_channels, dtype='float32')
        frame += 1
    return data_array_to_return


def exponential_features(timestamps, addresses, tau=0.005, nb_channels=64, **options):
    """Creates exponential feature vectors for events in an audio event stream.
    The function does not perform exponentiation on every channel for every event but updates the current feature
    from the previous feature in a recursive manner making use of the function update_feature.
    Args:
        :param timestamps: The timestamps in the event stream, a 1 dimensional numpy array of length nb_events.
        :param addresses: The addresses in the event stream, a 2 dimensional numpy array of addresses; the first column
        holds the channel address, while the second column holds the type of neuron.
        :param tau: The time constant for the exponential features.
        :param nb_channels: The number of frequency channels.
    Returns:
        The exponential features for the event stream.
    """
    exp_data = np.zeros(shape=(timestamps.shape[0], nb_channels), dtype='float32')
    feature = np.zeros(shape=nb_channels, dtype='float32')
    previous_time = timestamps[0]
    for idx, timestamp in enumerate(timestamps):
        feature = update_feature(feature, previous_time, timestamp, addresses[idx], tau)
        previous_time = timestamp
        exp_data[idx] = feature
    return exp_data


def update_feature(previous_feature, pts, cts, channel, tau=0.005):
    """Updates the exponential feature for current event given the previous feature.
    For every new event, the feature is just updated from the feature of the previous event. The value of the current
    feature at the current channel would be 1, while the value of the current feature at every other channel is just
    the corresponding value from the previous feature with a decay. The decay depends on the time elapsed between the
    previous event and the current event.
    Args:
        :param previous_feature: The feature of the previous event with shape (nb_channels). The actual shape of the
        feature is irrelevant, although the value of the parameter channel has to be less than nb_channels.
        :param pts: The time stamp of previous event.
        :param cts: The time stamp of current event.
        :param channel: The channel address of the current event.
        :type channel: int
        :param tau: The time constant for the exponential feature.
    Returns:
        The feature for the current event of the same shape as the parameter input_feature, (nb_channels).
    """
    time_elapsed = (cts - pts) / tau
    feature_to_return = previous_feature * np.exp(-1. * time_elapsed)
    feature_to_return[channel] = 1
    return feature_to_return


def filter_data(timestamps, addresses, filter_neuron=True, sort_time_stamps=True, filter_neuron_value=1,
                print_log=True, remove_double_spikes=False, remove_beeps=False, beep_length=0.1):
    if filter_neuron:
        unique_neurons = np.unique(addresses[:, 1])
        if filter_neuron_value not in unique_neurons:
            if print_log:
                print('The neuron type to be filtered was {}, but it is not in the data.'.format(filter_neuron_value))
            filter_neuron_value = np.random.choice(unique_neurons)
            if print_log:
                print('Rather, a neuron value of {} was chosen randomly which is present in the data.'
                      .format(filter_neuron_value))

        timestamps = timestamps[addresses[:, 1] == filter_neuron_value]
        addresses = addresses[addresses[:, 1] == filter_neuron_value]

    if remove_beeps:
        indices = np.logical_and(timestamps > timestamps[0] + beep_length, timestamps < timestamps[-1] - beep_length)
        addresses = addresses[indices]
        timestamps = timestamps[indices]

    if sort_time_stamps:
        timestamp_sort_permutation = np.argsort(timestamps)
        timestamps = timestamps[timestamp_sort_permutation]
        addresses = addresses[timestamp_sort_permutation]

    return timestamps, addresses