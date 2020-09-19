"""This module handles drift corrections."""


import numpy as np
import scipy.signal
import python_for_imscroll.image_processing as imp


def make_drift_list_simple(drift_fit):
    data = drift_fit['data'].item()
    n_aois = int(data[:, 0].max())
    start_frame = int(data[:, 1].min())
    end_frame = int(data[:, 1].max())
    n_frames = end_frame - start_frame + 1
    coords = data[:, 3:5]
    coords = coords.reshape((n_frames, n_aois, 2))
    diff_coords = np.diff(coords, axis=0, prepend=0)
    mean_diff_coords = np.mean(diff_coords, axis=1)
    mean_displacement = np.cumsum(mean_diff_coords, axis=0)
    filtered_displacement = scipy.signal.savgol_filter(mean_displacement,
                                                       window_length=11,
                                                       polyorder=2,
                                                       axis=0)
    driftlist = np.zeros((end_frame, 3))
    driftlist[:, 0] = np.arange(end_frame)
    driftlist[start_frame:end_frame+1, 1:] = np.diff(filtered_displacement, axis=0)
    return driftlist


class DriftCorrector:
    def __init__(self, driftlist):
        self._driftlist = driftlist

    def shift_aois(self, aois, frame):
        start_frame = aois.frame
        drift_range = np.logical_and(self._driftlist[:, 0] >= start_frame + 1,
                                     self._driftlist[:, 0] <= frame)
        drift = self._driftlist[drift_range, 1:].sum(axis=0)[np.newaxis, :]
        new_aois = imp.Aois(aois._coords + drift,
                            frame=frame,
                            width=aois.width,
                            frame_avg=aois.frame_avg,
                            channel=aois.channel)
        return new_aois
