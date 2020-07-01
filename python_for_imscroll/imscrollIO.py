#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (imscrollIO.py) is part of python_for_imscroll.
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

import sys
import json
from typing import List
from pathlib import Path
import scipy.io as sio
import numpy as np
import xarray as xr
from PySide2.QtWidgets import QApplication, QFileDialog, QWidget
from PySide2.QtCore import QTimer


def import_everything(filestr, datapath):
    data = initialize_data_from_intensity_traces(datapath, filestr)
    data = import_image_path_from_driftfit(data)
    data = import_time_stamps_for_proem(data)
    data = import_interval_results(data)
    data = import_viterbi_paths(data)
    return data


def def_data_path():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    w = QWidget()
    data_dir_str = QFileDialog.getExistingDirectory(w, caption='Select data path')
    datapath = Path(data_dir_str)
    # Let the event loop terminate after 1 ms and quit QApplication
    QTimer.singleShot(1, app.quit)
    app.exec_()
    return datapath


def qt_getfile():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    w = QWidget()
    data_dir_str = QFileDialog.getOpenFileName(w, caption='Select file')[0]
    datapath = Path(data_dir_str)
    # Let the event loop terminate after 1 ms and quit QApplication
    QTimer.singleShot(1, app.quit)
    app.exec_()
    return datapath


def qt_save_file():
    # There are still some problem need to solve!
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    w = QWidget()
    FILTERS = ('SVG Images (*.svg)', '.svg'
               'PNG Images (*.png)')
    filter_string = ''
    for file_filter in FILTERS:
        filter_string += (file_filter + ';;')
    filter_string = filter_string[:-2]

    file_path, selected_filter = QFileDialog.getSaveFileName(w, caption='Save to file',
                                            filter=filter_string)
    file_path = Path(file_path)
    # if file_path.suffix == '':
    #     if selected_filter ==
    #     file_path.with_suffix()
    # Let the event loop terminate after 1 ms and quit QApplication
    QTimer.singleShot(1, app.quit)
    app.exec_()
    return file_path


def import_channels_info(datapath, filestr):
    channels_info_file_path = datapath / (filestr + '_channels.dat')
    channels_info_file = sio.loadmat(channels_info_file_path)
    channels_info = channels_info_file['channelsInfo']
    channels_info_dict = dict()
    for i in range(channels_info.shape[0]):
        channels_info_dict[channels_info[i, 0].item()] = Path(channels_info[i, 1].item())

    return channels_info_dict


def import_time_stamps(channels_info, start_end):
    channels_time_stamps = dict(channels_info)
    timezeros = []
    for channel, image_path in channels_info.items():
        header_file_path = image_path / 'header.mat'
        header_file = sio.loadmat(header_file_path)
        time_stamps = header_file['vid']['ttb'][0, 0].squeeze()[slice(*start_end)]
        channels_time_stamps[channel] = time_stamps
        timezeros.append(time_stamps[0])
    starttime = min(timezeros)
    for channel, time_stamps in channels_time_stamps.items():
        channels_time_stamps[channel] = (time_stamps - starttime) / 1000

    return channels_time_stamps


def initialize_data_from_intensity_traces(datapath, filestr, start_end):
    channels_info = import_channels_info(datapath, filestr)
    channels_time_stamps = import_time_stamps(channels_info, start_end)

    intensity_file_path = datapath / (filestr + '_traces.dat')
    traces_file = sio.loadmat(intensity_file_path)
    channels = traces_file['traces'].dtype.names
    list_of_intensity_arrays = []
    nAOI = traces_file['traces'][channels[0]][0, 0].shape[0]
    for i_channel in channels:
        i_intensity = xr.DataArray(traces_file['traces'][i_channel][0, 0],
                                   dims=('AOI', 'time'),
                                   coords={'AOI': range(1, nAOI + 1),
                                           'time': channels_time_stamps[i_channel]})

        list_of_intensity_arrays.append(i_intensity)

    intensity: xr.DataArray = xr.concat(list_of_intensity_arrays, dim='channel')

    data = xr.Dataset({'intensity': (['channel', 'AOI', 'time'], intensity)},
                      coords={'AOI': intensity.AOI,
                              'time': intensity.time,
                              'channel': list(channels)})
    data = data.stack(channel_time=['channel', 'time'])
    data = data.dropna(dim='channel_time', how='all')
    data.attrs['datapath'] = datapath
    data.attrs['filestr'] = filestr

    return data


def import_image_path_from_driftfit(data):
    driftfit_file_path = data.datapath / (data.filestr + '_driftfit.dat')
    driftfit_file = sio.loadmat(driftfit_file_path)
    data.attrs['image_path'] = Path(driftfit_file['aoifits']['tifFile'][0][0][0])
    return data


def import_time_stamps_for_proem(data):
    time_stamp_txt = data.attrs['image_path'].with_suffix('.txt')
    time_data = np.loadtxt(time_stamp_txt)
    time_data[1, :] = np.cumsum(time_data[1, :])
    intcorrected_file_path = data.datapath / (data.filestr + '_intcorrected.dat')
    intcorrected_file = sio.loadmat(intcorrected_file_path)
    frames_used_in_intensity_cal = intcorrected_file['aoifits']['dataRed'][0, 0][:, 1]
    frame_start = int(frames_used_in_intensity_cal.min())
    frame_end = int(frames_used_in_intensity_cal.max())

    data.attrs['frame_start'] = frame_start
    data.attrs['frame_end'] = frame_end

    time_stamps = time_data[1, frame_start:frame_end + 1]
    data = data.assign_coords(time=time_stamps)

    return data


def import_interval_results(data):
    interval_file_path = data.datapath / (data.filestr + '_interval.dat')
    interval_file = sio.loadmat(interval_file_path)
    interval_traces_list = []
    for i_channel in data.binder_channel:
        i_interval_traces = np.zeros(data.intensity.sel(channel=i_channel).shape)
        for iAOI in range(0, data.intensity.shape[0]):
            i_interval_traces[iAOI, :] = \
                interval_file['intervals'][i_channel][0, 0]['AllTracesCellArray'][0, 0][iAOI, 0][:, 0]
        i_interval_traces = np.expand_dims(i_interval_traces, 2)
        i_interval_traces = xr.DataArray(i_interval_traces,
                                         dims=('AOI', 'time', 'channel'),
                                         coords={'AOI': range(1, data.intensity.shape[0] + 1),
                                                 'time': data.time.sel(channel=i_channel),
                                                 'channel': [i_channel]})
        i_interval_traces = i_interval_traces.stack(channel_time=['channel', 'time'])
        interval_traces_list.append(i_interval_traces)
    interval_traces = xr.concat(interval_traces_list, dim='channel_time')
    data['interval_traces'] = interval_traces
    return data


def import_viterbi_paths(data):
    data.attrs['datapath'] = Path(data.datapath)
    eb_file_path = data.datapath / (data.filestr + '_eb.dat')
    eb_file = sio.loadmat(eb_file_path)['eb_result']
    viterbi_path_list = []

    for i_channel in list(set(data.channel.values)):
        n_frames = len(data.time.sel(channel=i_channel))
        i_vit = np.zeros((len(data.AOI), n_frames, 2))
        for iAOI in range(0, len(data.AOI)):
            i_vit[iAOI, :, 0] = eb_file[i_channel][0, 0]['Vit'][0, 0][0, iAOI]['x'].squeeze()
            i_vit[iAOI, :, 1] = eb_file[i_channel][0, 0]['Vit'][0, 0][0, iAOI]['z'].squeeze()
        i_vit = np.expand_dims(i_vit, 3)
        i_viterbi_path = xr.DataArray(i_vit,
                                      dims=('AOI', 'time', 'state', 'channel'),
                                      coords={'channel': [i_channel],
                                              'state': ['position', 'label'],
                                              'AOI': data.AOI,
                                              'time': data.time.sel(channel=i_channel)})
        i_viterbi_path = i_viterbi_path.stack(channel_time=('channel', 'time'))
        viterbi_path_list.append(i_viterbi_path)
    data['viterbi_path'] = xr.concat(viterbi_path_list, dim='channel_time')
    return data


def save_data_to_netcdf(path, data):
    data.attrs['datapath'] = str(data.datapath)
    # data.attrs['image_path'] = str(data.image_path)
    data.to_netcdf(path)
    return 0


def load_data_from_netcdf(path):
    data = xr.open_dataset(path)
    data.attrs['datapath'] = Path(data.datapath)
    data.attrs['image_path'] = Path(data.image_path)
    return data


def save_data_to_json(path, data):
    data.attrs['datapath'] = str(data.datapath)
    data = data.reset_index('channel_time')
    data_dict = data.to_dict()
    with open(path, 'w') as file:
        json.dump(data_dict, file)
    return 0


def load_data_from_json(path):
    with open(path) as file:
        data_dict = json.load(file)
    data = xr.Dataset.from_dict(data_dict)
    data = data.set_index(channel_time=['channel', 'time'])
    # data.attrs['datapath'] = Path(data.datapath)
    return data


def get_xlsx_parameter_file_path() -> Path:
    """Opens a window to select the xlsx parameter file."""
    while True:
        xlsx_parameter_file_path = qt_getfile()
        print(xlsx_parameter_file_path)
        yes_no = input('Confirm [y/n]: ')
        if yes_no == 'y':
            xlsx_parameter_file_path = Path(xlsx_parameter_file_path)
            break
    return xlsx_parameter_file_path


def input_sheets_for_analysis() -> List[str]:
    """Request user input in console for list of sheets to be analyzed.

    The input format should be 'SHEET1, SHEET2, SHEET3, '..."""
    while True:
        input_str = input('Enter the sheets to be analyzed: ')
        sheet_list_out = input_str.split(', ')
        print(sheet_list_out)
        yes_no = input('Confirm [y/n]: ')
        if yes_no == 'y':
            break
    return sheet_list_out
