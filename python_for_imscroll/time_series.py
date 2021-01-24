from pathlib import Path
import numpy as np
import xarray as xr
from python_for_imscroll import binding_kinetics as bk


class TimeTraces():

    def __init__(self, n_traces, channels=None):
        if channels is None:
            self._data = None
        else:
            self._data = dict.fromkeys(channels.keys())
        self.n_traces = n_traces
        self._time = channels
        self.aoi_categories = None

    def get_n_traces(self):
        return len(self._data.AOI)

    def get_channels(self):
        return list(self._data.keys())

    @classmethod
    def from_xarray_json_all_file(cls, path):
        all_data, aoi_categories = bk.load_all_data(path)
        time_traces = cls(0)
        time_traces._data = all_data['data']
        time_traces.aoi_categories = aoi_categories
        return time_traces

    def get_time(self, channel, molecule):
        data = self._data.sel(channel=channel, AOI=molecule)
        return data.time.values

    def get_intensity(self, channel, molecule):
        data = self._data[channel].sel(molecule=molecule)
        return data.intensity.values

    def set_value(self, variable_name, channel, time, array):
        if len(array) != self.n_traces:
            raise ValueError(f'Input array length ({len(array)}) does '
                             f'not match n_traces ({self.n_traces}).')
        time_arr = self._time[channel]
        if self._data[channel] is None:
            data = {variable_name: (['molecule', 'time'], np.zeros((self.n_traces, len(time_arr))))}
            self._data[channel] = xr.Dataset(data,
                                             coords={'molecule': (['molecule'], np.arange(self.n_traces)),
                                                     'time': (['time'], time_arr)})
        elif variable_name not in self._data[channel].keys():
            self._data[channel] = self._data[channel].assign({variable_name: (['molecule', 'time'],
                                                                              np.zeros((self.n_traces, len(time_arr))))})
        self._data[channel][variable_name].loc[dict(time=time)] = array

    def get_state_mean_sequence(self, channel, molecule):
        data = self._data.sel(channel=channel, AOI=molecule)
        return data.viterbi_path.sel(state='position').values
