from __future__ import print_function

import os
import warnings
try:
    from tkFileDialog import askopenfilename
except:
    from tkinter.filedialog import askopenfilename
import numpy as np

NB_CHANNELS = 64
NB_NEURON_TYPES = 4
NB_EARS = 2
NB_FILTER_BANKS = 2
NB_ON_OFF = 2


# noinspection PyTypeChecker
def loadaerdat(filename=None, curr_directory=False, max_events=30000000):
    """Gets the event timestamps and the corresponding addresses for a .aedat or a .dat file.

    The function implements in python the matlab function loadaerdat.m;
    this function was written by Zhe He (zhhe@ini.uzh.ch).

    Args:
        :param filename: (optional) The path to the .aedat or .dat file;
        the path is relative to the current directory if curr_directory is True, else the path has to be absolute;
        if the filename is not given, then the user is asked through a dialog box to select the file himself.
        :param curr_directory: (optional) A boolean flag, if True the path in filename has to be relative to the
        current directory, else it has to be absolute.
        :param max_events: The maximum number of events to load from the file.

    Returns:
        :return: A tuple (timestamps, addresses).
        timestamps - A single dimensional numpy array holding the timestamps in microseconds, of length n_events.
        addresses - A single dimensional numpy array holding the corresponding address values, of length n_events.

    """
    if filename is None:
        filename = askopenfilename()
    elif curr_directory is True:
        curr_directory = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(curr_directory, filename)

    assert (filename.endswith('.aedat') or filename.endswith('.dat')), 'The given file has to be either a ' \
                                                                       '.aedat file or a .dat file'

    token = '#!AER-DAT'
    with open(filename, 'rb') as f:
        version = 2  # default version value
        for line in iter(f.readline, ''):
            try:
                line = line.decode("utf-8")
                if '#' in line:
                    if line[:len(token)] == token:
                        version = float(line[len(token):])
                    bof = f.tell()
                else:
                    break
            except Exception as e:
                break

        num_bytes_per_event = 6
        if version == 1:
            num_bytes_per_event = 6
        elif version == 2:
            num_bytes_per_event = 8
        f.seek(0, 2)
        eof = f.tell()
        # noinspection PyUnboundLocalVariable
        num_events = (eof - bof) // num_bytes_per_event
        if num_events > max_events:
            num_events = max_events

        f.seek(bof)
        if version == 2:
            data = np.fromfile(f, dtype='>u4', count=2 * num_events)
            all_address = data[::2]
            all_timestamps = data[1::2]
        elif version == 1:
            data = np.fromfile(f, dtype='>u2', count=3 * num_events)
            all_address = data[::3]
            data_time_stamps = np.delete(data, slice(0, data.shape[0], 3))
            all_timestamps = np.fromstring(data_time_stamps.tobytes(), dtype='>u4', count=num_events)
        else:
            warnings.warn("The AER-DAT version of the current file is {}, "
                          "the loading function for which has not been implemented yet.".format(version))

    # noinspection PyUnboundLocalVariable
    return all_timestamps.astype(np.uint32), all_address.astype(np.uint32)


# noinspection PyTypeChecker
def decode_ams1b(timestamps, addresses, return_type=True, reset_time_stamps=True):
    time_wrap_mask = int("80000000", 16)
    adc_event_mask = int("2000", 16)

    address_mask = int("00FC", 16)
    neuron_mask = int("0300", 16)
    ear_mask = int("0002", 16)
    filterbank_mask = int("0001", 16)

    # temporarily remove all StartOfConversion events, no clue what this means!
    soc_idx_1 = int("302C", 16)
    soc_idx_2 = int("302D", 16)
    events_without_start_of_conversion_idx = np.where(np.logical_and(addresses != soc_idx_1, addresses != soc_idx_2))
    timestamps = timestamps[events_without_start_of_conversion_idx]
    addresses = addresses[events_without_start_of_conversion_idx]

    # finding cochlea events
    cochlea_events_idx = np.where(np.logical_and(addresses & time_wrap_mask == 0, addresses & adc_event_mask == 0))
    timestamps_cochlea = timestamps[cochlea_events_idx]
    addresses_cochlea = addresses[cochlea_events_idx]

    if reset_time_stamps:
        timestamps_cochlea = timestamps_cochlea - timestamps_cochlea[0]
    timestamps_cochlea = timestamps_cochlea.astype(np.float32) / 1e6

    # decoding addresses to get ear id, on off id and the channel id
    channel_id = np.array((addresses_cochlea & address_mask) >> 2, dtype=np.int8)
    ear_id = np.array((addresses_cochlea & ear_mask) >> 1, dtype=np.int8)
    neuron_id = np.array((addresses_cochlea & neuron_mask) >> 8, dtype=np.int8)
    filterbank_id = np.array((addresses_cochlea & filterbank_mask), dtype=np.int8)

    if return_type:
        type_id = get_type_id('ams1b', channel=channel_id, neuron=neuron_id, filterbank=filterbank_id)
        return timestamps_cochlea, ear_id, type_id

    return timestamps_cochlea, channel_id, ear_id, neuron_id, filterbank_id


# noinspection PyTypeChecker
def decode_lp(timestamps, addresses, return_type=True):
    time_wrap_mask = int("80000000", 16)
    adc_event_mask = int("2000", 16)

    address_mask = int("00FC", 16)
    on_off_mask = int("0001", 16)
    ear_mask = int("0002", 16)

    # temporarily remove all StartOfConversion events, no clue what this means!
    soc_idx_1 = int("302C", 16)
    soc_idx_2 = int("302D", 16)
    events_without_start_of_conversion_idx = np.where(np.logical_and(addresses != soc_idx_1, addresses != soc_idx_2))
    timestamps = timestamps[events_without_start_of_conversion_idx]
    addresses = addresses[events_without_start_of_conversion_idx]

    # finding cochlea events
    cochlea_events_idx = np.where(np.logical_and(addresses & time_wrap_mask == 0, addresses & adc_event_mask == 0))
    timestamps_cochlea = timestamps[cochlea_events_idx]
    addresses_cochlea = addresses[cochlea_events_idx]

    timestamps_cochlea = timestamps_cochlea - timestamps_cochlea[0]
    timestamps_cochlea = timestamps_cochlea.astype(np.float32) / 1e6

    # decoding addresses to get ear id, on off id and the channel id
    channel_id = np.array((addresses_cochlea & address_mask) >> 2, dtype=np.int8)
    ear_id = np.array((addresses_cochlea & ear_mask) >> 1, dtype=np.int8)
    on_off_id = np.array((addresses_cochlea & on_off_mask), dtype=np.int8)

    if return_type:
        type_id = get_type_id('lp', channel=channel_id, on_off=on_off_id)
        return timestamps_cochlea, ear_id, type_id

    return timestamps_cochlea, channel_id, ear_id, on_off_id


# noinspection PyTypeChecker
def decode_ams1c(timestamps, addresses, return_type=True):
    time_wrap_mask = int("80000000", 16)
    adc_event_mask = int("2000", 16)

    address_mask = int("00FC", 16)
    neuron_mask = int("0300", 16)
    ear_mask = int("0002", 16)
    filterbank_mask = int("0001", 16)

    # temporarily remove all StartOfConversion events, no clue what this means!
    soc_idx_1 = int("302C", 16)
    soc_idx_2 = int("302D", 16)
    events_without_start_of_conversion_idx = np.where(np.logical_and(addresses != soc_idx_1, addresses != soc_idx_2))
    timestamps = timestamps[events_without_start_of_conversion_idx]
    addresses = addresses[events_without_start_of_conversion_idx]

    # finding cochlea events
    cochlea_events_idx = np.where(np.logical_and(addresses & time_wrap_mask == 0, addresses & adc_event_mask == 0))
    timestamps_cochlea = timestamps[cochlea_events_idx]
    addresses_cochlea = addresses[cochlea_events_idx]

    timestamps_cochlea = timestamps_cochlea - timestamps_cochlea[0]
    timestamps_cochlea = timestamps_cochlea.astype(np.float32) / 1e6

    # decoding addresses to get ear id, on off id and the channel id
    channel_id = np.array((addresses_cochlea & address_mask) >> 2, dtype=np.int8)
    ear_id = np.array((addresses_cochlea & ear_mask) >> 1, dtype=np.int8)
    neuron_id = np.array((addresses_cochlea & neuron_mask) >> 8, dtype=np.int8)
    filterbank_id = np.array((addresses_cochlea & filterbank_mask), dtype=np.int8)

    if return_type:
        type_id = get_type_id('ams1b', channel=channel_id, neuron=neuron_id, filterbank=filterbank_id)
        return timestamps_cochlea, ear_id, type_id

    return timestamps_cochlea, channel_id, ear_id, neuron_id, filterbank_id


def get_type_id(sensor_type, channel=None, neuron=None, filterbank=None, on_off=None):
    if sensor_type == 'ams1b' or sensor_type == 'ams1c':
        type_id = channel + NB_CHANNELS * neuron + NB_CHANNELS * NB_NEURON_TYPES * filterbank
        return type_id
    elif sensor_type == 'lp':
        type_id = channel + NB_CHANNELS * on_off
        return type_id
    else:
        warnings.warn('The sensor type is not implemented yet.')


def separate_streams(timestamps, itds, itd_indices, num_streams=7):
    itd_streams = []
    timestamps = timestamps[itd_indices]
    min_timestamp, max_timestamp = np.amin(timestamps), np.amax(timestamps)
    time_length = (max_timestamp - min_timestamp) / num_streams
    for idx in range(num_streams):
        indices = np.where((timestamps > min_timestamp + idx * time_length) &
                           (timestamps < min_timestamp + (idx + 1) * time_length))[0]
        itds_to_append = itds[indices]
        itd_streams.append(itds_to_append)
    return itd_streams


def get_labels(timestamps, itd_indices, num_streams=7):
    timestamps = timestamps[itd_indices]
    labels = np.zeros_like(timestamps)
    min_timestamp, max_timestamp = np.amin(timestamps), np.amax(timestamps)
    time_length = (max_timestamp - min_timestamp) / num_streams
    for idx in range(num_streams):
        indices = np.where((timestamps > min_timestamp + idx * time_length) &
                           (timestamps < min_timestamp + (idx + 1) * time_length))[0]
        labels[indices] = idx
    return labels

if __name__ == '__main__':
    test_timestamps, test_addresses = loadaerdat()
    test_timestamps, test_ears, test_types = decode_ams1b(test_timestamps, test_addresses)
    print('Done')
