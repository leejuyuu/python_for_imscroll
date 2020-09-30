"""
This module handles mapping operations, which is the image registration process
done via transforming AOI coordinates.
"""
from collections import namedtuple
import re
import numpy as np
import scipy.io as sio
import python_for_imscroll.image_processing as imp

MapDirection = namedtuple('MapDirection', ['from_channel', 'to_channel'])
DIR_DICT = {'r': 'red',
            'g': 'green',
            'b': 'blue'}
AVAILABLE_CHANNELS = ('red', 'green', 'blue')

class Mapper():
    def __init__(self):
        self.map_matrix = None

    @classmethod
    def from_imscroll(cls, path):
        mapper = cls()
        file_name = str(path.stem)
        match = re.search('_[rgb]{2}', file_name)
        if match:
            direction_str = file_name[match.start()+1:match.end()]
            direction = MapDirection(*(DIR_DICT[i] for i in direction_str))
        else:
            raise ValueError('Mapping file name does not provide direction information.')
        mapper.map_matrix = dict()
        mapping_points = sio.loadmat(path)['mappingpoints']
        mapper.map_matrix[direction] = make_map_matrix(mapping_points[:, [3, 2]]-1,
                                                       mapping_points[:, [9, 8]]-1)
        return mapper

    def map(self, aois, to_channel: str):
        if aois.channel not in AVAILABLE_CHANNELS:
            raise ValueError('From-channel is not one of the available channels')
        if to_channel not in AVAILABLE_CHANNELS:
            raise ValueError('To-channel is not one of the available channels')
        if aois.channel == to_channel:
            aois_copy = imp.Aois(aois._coords,
                                 frame=aois.frame,
                                 frame_avg=aois.frame_avg,
                                 width=aois.width,
                                 channel=to_channel)
            return aois_copy

        direction = MapDirection(from_channel=aois.channel,
                                 to_channel=to_channel)
        inv_direction = MapDirection(from_channel=to_channel,
                                     to_channel=aois.channel)
        if direction in self.map_matrix:
            map_matrix = self.map_matrix[direction]
        elif inv_direction in self.map_matrix:
            map_matrix = self._inverse_map_matrix(self.map_matrix[inv_direction])
        else:
            raise ValueError(f'Mapping matrix from channel {aois.channel}'
                             ' to channel {to_channel} is not loaded')

        new_coords = np.matmul(map_matrix[:, :2], aois._coords.T) + map_matrix[:, 2, np.newaxis]
        mapped_aois = imp.Aois(new_coords.T,
                               frame=aois.frame,
                               frame_avg=aois.frame_avg,
                               width=aois.width,
                               channel=to_channel)
        return mapped_aois

    @staticmethod
    def _inverse_map_matrix(map_matrix):
        inv_A = np.linalg.inv(map_matrix[:, :2])
        inv_b = np.matmul(-inv_A, map_matrix[:, 2, np.newaxis])
        inv_map_matrix = np.concatenate((inv_A, inv_b), axis=1)
        return inv_map_matrix


def make_map_matrix(x1y1, x2y2):
    map_matrix = np.vstack((double_linear_regression(x2y2[:, 0], x1y1),
                            double_linear_regression(x2y2[:, 1], x1y1)))
    return map_matrix


def double_linear_regression(y, x1x2):
    X = np.hstack((x1x2, np.ones((x1x2.shape[0], 1))))
    beta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, y))
    return beta.squeeze()
