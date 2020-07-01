#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (binding_kinetics.py) is part of python_for_imscroll.
#
#  python_for_imscroll is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  python_for_imscroll is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with python_for_imscroll.  If not, see <https://www.gnu.org/licenses/>.

import json
from collections import namedtuple
import xarray as xr
import numpy as np


def find_state_intensity_and_std(channel_time_series):
    """Return intensity mean and standard deviation in namedtuple, nested in AOI > states order"""
    number_of_states_array = get_number_of_states(channel_time_series)
    mean_std_pair = namedtuple('mean_std_pair', ('mean', 'std'))
    intensity_pairs = []
    for i_aoi in channel_time_series.AOI:
        number_of_states = number_of_states_array.sel(AOI=i_aoi)
        viterbi_intensity = channel_time_series['viterbi_path'].sel(state='position', AOI=i_aoi)
        viterbi_state_labels = channel_time_series['viterbi_path'].sel(state='label', AOI=i_aoi)
        raw_intensity = channel_time_series['intensity'].sel(AOI=i_aoi)
        i_aoi_pairs = []
        for state_label in range(1, int(number_of_states) + 1):
            state_viterbi_intensity = viterbi_intensity[viterbi_state_labels == state_label]
            state_unique_intensity = np.unique(state_viterbi_intensity)
            if len(state_unique_intensity) != 1:
                raise ValueError('State intensity is not unique.')
            state_raw_intensity = raw_intensity[viterbi_state_labels == state_label]
            intensity_std = np.std(state_raw_intensity)
            i_aoi_pairs.append(mean_std_pair(state_unique_intensity.item(), intensity_std.item()))
        intensity_pairs.append(tuple(i_aoi_pairs))

    return intensity_pairs


def find_state_with_lowest_intensity(state_intensity_pairs):
    mean_intensity_list = []
    for state_intensity_pair in state_intensity_pairs:
        mean_intensity_list.append(state_intensity_pair.mean)
    lowest_state_index = np.argmin(mean_intensity_list)
    return lowest_state_index


def check_if_state_equal_to_zero(state_intensity):
    is_state_equal_to_zero = abs(state_intensity.mean) <= 2 * state_intensity.std
    return is_state_equal_to_zero


def get_number_of_states(channel_data):
    viterbi_state_labels = channel_data['viterbi_path'].sel(state='label')
    number_of_states = viterbi_state_labels.max(dim='time')

    return number_of_states


def list_aois_more_than_two_states(nStates):
    """Returns a list containing aois that visits more than 2 states."""
    # np returns zero index so plus 1
    aoi_list = (np.where(nStates > 2)[0] + 1).tolist()
    return aoi_list


def list_aois_elevated_two_states(channel_state_info):
    """Returns a list containing aois that visits only 2 states
    but the lowest state is not at baseline 0, which indicates multiple tether."""
    condition = ((channel_state_info.nStates == 2)
                 & np.logical_not(channel_state_info.bool_lowest_state_equal_to_zero))
    aoi_list = channel_state_info.AOI[condition].values.tolist()
    return aoi_list


def collect_channel_state_info(channel_time_series):
    intensity_pairs = find_state_intensity_and_std(channel_time_series)
    lowest_state_label = []
    is_lowest_state_equal_to_zero_list = []
    for i_aoi_pairs in intensity_pairs:
        lowest_state_index = find_state_with_lowest_intensity(i_aoi_pairs)
        is_lowest_state_equal_zero = check_if_state_equal_to_zero(i_aoi_pairs[lowest_state_index])
        lowest_state_label.append(lowest_state_index + 1)
        is_lowest_state_equal_to_zero_list.append(is_lowest_state_equal_zero)
    nStates = get_number_of_states(channel_time_series)
    channel_state_info = xr.Dataset({'nStates': (['AOI'], nStates),
                                     'lowest_state_label': (['AOI'], lowest_state_label),
                                     'bool_lowest_state_equal_to_zero': (['AOI'], is_lowest_state_equal_to_zero_list)},
                                    coords={'AOI': nStates.AOI})
    return channel_state_info


def collect_all_channel_state_info(data):
    channel_data_list = []
    for i_channel in list(set(data.channel.values)):
        i_channel_state_info = collect_channel_state_info(get_channel_data(data, i_channel))
        i_channel_state_info = i_channel_state_info.assign_coords(channel=i_channel)

        channel_data_list.append(i_channel_state_info)
    channel_state_info = xr.concat(channel_data_list, dim='channel')

    return channel_state_info


def list_multiple_tethers(channel_state_info) -> tuple:
    bad_tethers1 = list_aois_more_than_two_states(channel_state_info.nStates)
    bad_tethers2 = list_aois_elevated_two_states(channel_state_info)
    bad_tethers = list(set(bad_tethers1) | set(bad_tethers2))
    bad_tethers.sort()
    bad_tethers = tuple(bad_tethers)
    return bad_tethers


def split_data_set_by_specifying_aoi_subset(data, aoi_subset: set):
    whole_aoi_set = set(data.AOI.values)
    complement_subset = whole_aoi_set - aoi_subset
    data_subset = data.sel(AOI=list(aoi_subset))
    data_complement = data.sel(AOI=list(complement_subset))
    return data_subset, data_complement


def colocalization_analysis(channel_data):
    channel_state_info = collect_channel_state_info(channel_data)
    channel_data = channel_data.merge(channel_state_info)
    bad_aoi_list = []
    intervals_list = []
    for iAOI in channel_data.AOI:
        # print(iAOI)
        state_sequence = channel_data['viterbi_path'].sel(state='label', AOI=iAOI)
        state_start_index = find_state_end_point(state_sequence)
        event_time = assign_event_time(state_sequence, state_start_index)
        i_intervals = set_up_intervals(event_time)
        i_intervals = assign_state_number_to_intervals(channel_data.sel(AOI=iAOI), i_intervals)
        intervals_list.append(i_intervals)
        if find_any_bad_intervals(channel_data.sel(AOI=iAOI), i_intervals):
            bad_aoi_list.append(int(iAOI.values))
    bad_aoi_list.sort()
    bad_aois = tuple(bad_aoi_list)
    intervals = xr.concat(intervals_list, dim='AOI')
    intervals = intervals.assign_coords(AOI=channel_data.AOI)
    intervals = intervals.assign_coords(channel=channel_data.channel.item())

    return bad_aois, intervals


def find_state_end_point(state_sequence):
    change_array = state_sequence.diff(dim='time')
    state_start_index = np.nonzero(change_array.values)[0]
    return state_start_index


def assign_event_time(state_sequence, state_end_index):
    time_for_each_frame = state_sequence.time
    event_time = np.zeros((len(state_end_index) + 2))
    event_time[0] = time_for_each_frame[0]
    event_time[-1] = time_for_each_frame[-1]
    # Assign the time point for events as the mid-point between two points that have different
    # state labels
    for i, i_end_index in enumerate(state_end_index):
        event_time[i + 1] = (time_for_each_frame[i_end_index] +
                             time_for_each_frame[i_end_index + 1]) / 2
    return event_time


def set_up_intervals(event_time):
    intervals = xr.Dataset({'duration': (['AOI', 'interval_number'], np.zeros((1, len(event_time) - 1)))},
                           coords={'interval_number': range(len(event_time) - 1)
                                   })
    intervals['duration'] = xr.DataArray(np.diff(event_time),
                                         dims='interval_number',
                                         coords={'interval_number': intervals.interval_number})
    intervals['start'] = xr.DataArray(event_time[0:-1], dims='interval_number')
    intervals['end'] = xr.DataArray(event_time[1:], dims='interval_number')
    return intervals


def shift_state_number(AOI_data):
    if AOI_data['bool_lowest_state_equal_to_zero']:
        if AOI_data['lowest_state_label'] == 1:
            AOI_data['viterbi_path'].loc[dict(state='label')] += -1
        else:
            raise ValueError('shift_state_number:\nlowest state not equal to 1')
    return AOI_data


def assign_state_number_to_intervals(AOI_data, intervals):
    AOI_data = shift_state_number(AOI_data)
    intervals_state_number = xr.DataArray(np.zeros(len(intervals.interval_number)),
                                          dims='interval_number',
                                          coords={'interval_number': intervals.interval_number})
    for i in intervals['interval_number']:
        interval_slice = slice(intervals['start'].loc[i], intervals['end'].loc[i])
        distinct_state_numbers = set(AOI_data['viterbi_path'].sel(state='label',
                                                                  time=interval_slice).values)
        if len(distinct_state_numbers) == 1:
            intervals_state_number.loc[i] = list(distinct_state_numbers)[0]

        else:
            raise ValueError("""assign_state_number_to_intervals:
            There are more than one state in this interval.""")
    intervals['state_number'] = intervals_state_number
    return intervals


def find_any_bad_intervals(AOI_data, intervals):
    any_bad_interval = False
    for i in intervals['interval_number']:
        interval_slice = slice(intervals['start'].loc[i], intervals['end'].loc[i])
        chunk_of_interval_traces = AOI_data['interval_traces'].sel(time=interval_slice)
        # interval trace: (-2, 0, 2)=on, (-1,1, 3)=off
        num_of_colocalized_time_points = np.count_nonzero(chunk_of_interval_traces.values % 2 != 0)

        # Viterbi on state must correspond to colocalized event, with 10% tolerance
        reject_condition_1 = intervals['state_number'].loc[i] != 0
        reject_condition_2 = num_of_colocalized_time_points < 0.9 * len(chunk_of_interval_traces)
        if reject_condition_1 and reject_condition_2:
            any_bad_interval = True
            break
    return any_bad_interval


def group_analyzable_aois_into_state_number(intervals):
    analyzable_tethers = {}
    for i_state in range(int(intervals['state_number'].max()) + 1):
        i_index = intervals['state_number'].max(dim='interval_number') == i_state
        analyzable_tethers[i_state] = i_index.AOI.where(i_index, drop=True).values.tolist()
    return analyzable_tethers


def extract_dwell_time(intervals_list, state):

    out_list = []
    for i_file_intervals in intervals_list:
        for iAOI in i_file_intervals.AOI:
            i_AOI_intervals = i_file_intervals.sel(AOI=iAOI)
            valid_intervals = i_AOI_intervals.where(
                np.logical_not(np.isnan(i_AOI_intervals.duration)),
                drop=True)

            # valid_intervals = valid_intervals.isel(interval_number=slice(1, None))
            if len(valid_intervals.duration) != 0:
                valid_intervals = valid_intervals.assign({'event_observed' :('interval_number', np.ones(len(valid_intervals.interval_number)))})

                valid_intervals['event_observed'][[0, -1]] = 0
                i_dwell = valid_intervals[['duration', 'event_observed']].where(valid_intervals.state_number == state,
                                                                                drop=True)

                i_dwell['event_observed'] = i_dwell['event_observed'].astype(bool)
                i_dwell = i_dwell.reset_index('interval_number')


                out_list.append(i_dwell)

    out = xr.concat(out_list, dim='interval_number')

    return out


def extract_first_binding_time(intervals_list):
    i_file_intervals = intervals_list[0]
    first_intervals = i_file_intervals.sel(interval_number=0)
    left_censored = first_intervals.state_number == 1        
    dwells = first_intervals['duration'].to_dataset()
    dwells['duration'] = xr.where(left_censored, 0, dwells['duration'])
    dwells['event_observed'] = np.logical_not(left_censored)
    return dwells


def get_channel_data(data, channel):
    channel_data = data.sel(channel=channel)
    channel_data = channel_data.assign_coords(channel=[channel])
    return channel_data


def save_all_data(all_data, AOI_categories, path):
    for key, value in all_data.items():
        if key == 'data':
            all_data[key] = value.reset_index('channel_time').to_dict()
        else:
            all_data[key] = value.to_dict()
    collected_data = {'all_data': all_data,
                      'AOI_categories': AOI_categories}
    with open(path, 'w') as file:
        json.dump(collected_data, file)
    return 0


def load_all_data(path):
    with open(path) as file:
        collected_data = json.load(file)
    all_data = collected_data['all_data']
    AOI_categories = collected_data['AOI_categories']
    for key, item in all_data.items():
        if key == 'data':
            all_data[key] = xr.Dataset.from_dict(item).set_index(channel_time=('channel', 'time'))
        else:
            all_data[key] = xr.Dataset.from_dict(item)
    return all_data, AOI_categories
