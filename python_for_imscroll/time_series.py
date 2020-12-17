from pathlib import Path
from python_for_imscroll import binding_kinetics as bk


class TimeTraces():

    def __init__(self):
        self.data = None
        self.aoi_categories = None

    def get_n_traces(self):
        return len(self.data.AOI)

    def get_channels(self):
        channels = [self.data.target_channel] + self.data.binder_channel
        return channels

    @classmethod
    def from_xarray_json_all_file(cls, path):
        all_data, aoi_categories = bk.load_all_data(path)
        time_traces = cls()
        time_traces.data = all_data['data']
        time_traces.aoi_categories = aoi_categories
        return time_traces

    def get_time(self, channel, molecule):
        data = self.data.sel(channel=channel, AOI=molecule)
        return data.time.values

    def get_intensity(self, channel, molecule):
        data = self.data.sel(channel=channel, AOI=molecule)
        return data.intensity.values

    def get_state_mean_sequence(self, channel, molecule):
        data = self.data.sel(channel=channel, AOI=molecule)
        return data.viterbi_path.sel(state='position').values
