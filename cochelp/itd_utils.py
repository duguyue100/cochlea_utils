from __future__ import division

import numpy as np
import time

import progressbar


def get_itds(timestamps, ears, types, max_itd=800e-6, save_to_file=None, verbose=False):
    ears = ears.astype(np.bool)
    itds_to_return = np.zeros(timestamps.size, dtype=np.float32)
    itds_to_return.fill(-5. * max_itd)

    timestamps_dict = {}
    timestamp_indices_dict = {}
    for ear in np.unique(ears):
        timestamps_dict[ear] = {}
        timestamp_indices_dict[ear] = {}
        for type_of_event in np.unique(types):
            timestamps_dict[ear][type_of_event] = []
            timestamp_indices_dict[ear][type_of_event] = []

    for idx, (timestamp, ear, type_of_event) in enumerate(zip(timestamps, ears, types)):
        timestamps_dict[ear][type_of_event].append(timestamp)
        timestamp_indices_dict[ear][type_of_event].append(idx)

    if verbose:
        print('Initialized the timestamp lists.')

    bar = progressbar.ProgressBar() if verbose else lambda x: x

    for type_of_event in bar(np.unique(types)):
        timestamps_left = np.array(timestamps_dict[True][type_of_event])
        timestamp_indices_left = timestamp_indices_dict[True][type_of_event]
        timestamps_right = np.array(timestamps_dict[False][type_of_event])
        timestamp_indices_right = timestamp_indices_dict[False][type_of_event]

        for ts_right, ts_idx_right in zip(timestamps_right, timestamp_indices_right):
            matched_indices = np.where((timestamps_left >= ts_right - max_itd) &
                                       (timestamps_left < ts_right + max_itd))[0]
            if matched_indices.size > 0:
                matched_itds = ts_right - timestamps_left[matched_indices]
                min_itd = np.argmin(np.abs(matched_itds))
                itds_to_return[ts_idx_right] = matched_itds[min_itd]

        for ts_left, ts_idx_left in zip(timestamps_left, timestamp_indices_left):
            matched_indices = np.where((timestamps_right >= ts_left - max_itd) &
                                       (timestamps_right < ts_left + max_itd))[0]
            if matched_indices.size > 0:
                matched_itds = timestamps_right[matched_indices] - ts_left
                min_itd = np.argmin(np.abs(matched_itds))
                itds_to_return[ts_idx_left] = matched_itds[min_itd]

    itd_indices = np.where(itds_to_return > -4. * max_itd)[0]
    itds_to_return = itds_to_return[itd_indices]
    if save_to_file is not None:
        np.savez(save_to_file, timestamps=timestamps, ears=ears, types=types, itds=itds_to_return,
                 itd_indices=itd_indices)

    return itds_to_return, itd_indices


def get_itd_dict(max_itd, num_bins):
    bin_length = 2 * max_itd / num_bins
    return np.array([-max_itd + (idx + 0.5) * bin_length for idx in range(num_bins)], dtype=np.float32)


def get_itds_v2(timestamps, ears, types, max_itd=800e-6, save_to_file=None, verbose=False):
    ears = ears.astype(np.bool)
    itds_to_return, timestamps_to_return, ears_to_return, types_to_return = [], [], [], []

    timestamps_dict = {}
    timestamp_indices_dict = {}
    for ear in np.unique(ears):
        timestamps_dict[ear] = {}
        timestamp_indices_dict[ear] = {}
        for type_of_event in np.unique(types):
            timestamps_dict[ear][type_of_event] = []
            timestamp_indices_dict[ear][type_of_event] = []

    for idx, (timestamp, ear, type_of_event) in enumerate(zip(timestamps, ears, types)):
        timestamps_dict[ear][type_of_event].append(timestamp)
        timestamp_indices_dict[ear][type_of_event].append(idx)

    if verbose:
        print('Initialized the timestamp lists.')

    bar = progressbar.ProgressBar() if verbose else lambda x: x

    for type_of_event in bar(np.unique(types)):
        timestamps_left = np.array(timestamps_dict[True][type_of_event])
        timestamp_indices_left = timestamp_indices_dict[True][type_of_event]
        timestamps_right = np.array(timestamps_dict[False][type_of_event])
        timestamp_indices_right = timestamp_indices_dict[False][type_of_event]

        for ts_right, ts_idx_right in zip(timestamps_right, timestamp_indices_right):
            matched_indices = np.where((timestamps_left >= ts_right - max_itd) &
                                       (timestamps_left < ts_right + max_itd))[0]
            for matched_index in matched_indices:
                matched_itd = ts_right - timestamps_left[matched_index]
                itds_to_return.append(matched_itd)
                timestamps_to_return.append(ts_right)
                ears_to_return.append(False)
                types_to_return.append(type_of_event)

        for ts_left, ts_idx_left in zip(timestamps_left, timestamp_indices_left):
            matched_indices = np.where((timestamps_right >= ts_left - max_itd) &
                                       (timestamps_right < ts_left + max_itd))[0]
            for matched_index in matched_indices:
                matched_itd = timestamps_right[matched_index] - ts_left
                itds_to_return.append(matched_itd)
                timestamps_to_return.append(ts_left)
                ears_to_return.append(True)
                types_to_return.append(type_of_event)

    indices = np.argsort(timestamps_to_return)
    timestamps_to_return = np.array(timestamps_to_return, dtype=np.float32)[indices]
    itds_to_return = np.array(itds_to_return, dtype=np.float32)[indices]
    types_to_return = np.array(types_to_return, dtype=np.int16)[indices]
    ears_to_return = np.array(ears_to_return, dtype=np.int8)[indices]

    if save_to_file is not None:
        np.savez(save_to_file, timestamps=timestamps_to_return, ears=ears_to_return,
                 types=types_to_return, itds=itds_to_return)

    return itds_to_return


if __name__ == '__main__':
    import es_utils as es

    test_timestamps, test_addresses = es.loadaerdat('../data/man_clean.aedat')
    test_timestamps, test_ears, test_types = es.decode_ams1b(test_timestamps, test_addresses)

    start = time.time()
    test_itds_v1 = get_itds(test_timestamps, test_ears, test_types, save_to_file='man_clean', verbose=True)
    test_itds_v2 = get_itds_v2(test_timestamps, test_ears, test_types, save_to_file='man_clean', verbose=True)
    print('Computing the itds complete, took {} seconds'.format(time.time() - start))
