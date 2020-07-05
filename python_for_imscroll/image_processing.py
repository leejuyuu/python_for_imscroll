from pathlib import Path
import math
from typing import Union
import scipy.io as sio
import scipy.ndimage
import numpy as np



class ImageSequence:
    def __init__(self, image_path):
        header_file_path = image_path / 'header.mat'

        header_file = sio.loadmat(header_file_path)
        self._offset = header_file['vid']['offset'][0, 0].squeeze()
        self._file_number = header_file['vid']['filenumber'][0, 0].squeeze()
        self.width = header_file['vid']['width'][0, 0].item()
        self.height = header_file['vid']['height'][0, 0].item()
        self.length = header_file['vid']['nframes'][0, 0].item()
        self._image_path = image_path

    def get_one_frame(self, frame: int):
        """
        Return one frame image in the sequence.
        """
        if not (isinstance(frame, int) and frame >= 0):
            raise ValueError('Frame number must be positive integers or 0.')
        if frame >= self.length:
            err_str = 'Frame number ({}) exceeds sequence length - 1 ({})'.format(self.length,
                                                                                  self.length - 1)
            raise ValueError(err_str)

        file_number = self._file_number[frame]
        offset = self._offset[frame]
        image_file_path = self._image_path / '{}.glimpse'.format(file_number)
        image = np.fromfile(image_file_path,
                            dtype='>i2',
                            count=(self.width * self.height),
                            offset=offset)
        image = np.reshape(image, (self.height, self.width))
        # The glimpse saved image is U16 integer - 2**15 and saved in I16 format.
        # To recover the U16 integers, we need to add 2**15 back, but we cannot
        # do this directly since the image is read as I16 integer, adding 2**15
        # will cause overflow. Need to cast to larger container (at least 32 bit)
        # first.
        image = image.astype(int) + 2**15
        return image

    def __iter__(self):
        return (self.get_one_frame(frame) for frame in range(self.length))

    def get_whole_stack(self):
        return np.stack([self.get_one_frame(frame) for frame in range(self.length)],
                        axis=-1)


def conv2(v1, v2, m, mode='same'):
    """
    Two-dimensional convolution of matrix m by vectors v1 and v2

    First convolves each column of 'm' with the vector 'v1'
    and then it convolves each row of the result with the vector 'v2'.

    """
    tmp = np.apply_along_axis(np.convolve, 0, m, v1, mode)
    return np.apply_along_axis(np.convolve, 1, tmp, v2, mode)

def band_pass(image: np.ndarray, r_noise, r_object):
    def normalize(x):
        return x/sum(x)

    if r_noise:
        gauss_kernel_size = np.ceil(r_noise * 5)
        x = np.arange(-gauss_kernel_size, gauss_kernel_size+1)
        print(x)
        gaussian_kernel = normalize(np.exp(-(x/r_noise/2)**2))
    else:
        gaussian_kernel = 1
    print(gaussian_kernel)
    print(sum(gaussian_kernel))
    print(np.arange(10)**2)
    gauss_filtered_image = conv2(gaussian_kernel, gaussian_kernel, image)

    if r_object:
        boxcar_kernel = normalize(np.ones(round(r_object)*2 + 1))
        boxcar_image = conv2(boxcar_kernel, boxcar_kernel, image)
        band_passed_image = gauss_filtered_image - boxcar_image
    else:
        band_passed_image = gauss_filtered_image

    edge_size = round(max(r_object, gauss_kernel_size))
    band_passed_image[0:edge_size, :] = 0
    band_passed_image[-edge_size:, :] = 0
    band_passed_image[:, 0:edge_size] = 0
    band_passed_image[:, -edge_size:] = 0
    band_passed_image[band_passed_image<0] = 0
    return band_passed_image


def find_peaks(image, threshold, peak_size):
    # Transpose the image to mimick the matlab index ordering, since the later
    # close peak removal is dependent on the order of the peaks.
    t_image = image.T
    is_above_th = t_image > threshold
    if not is_above_th.any():
        print('nothing above threshold')
        return None
    width, height = t_image.shape

    is_local_max = scipy.ndimage.maximum_filter(t_image, size=3) == t_image
    idx = np.logical_and(is_above_th, is_local_max)  # np.nonzero returns the indices
    # The index is labeled as x, y since the image was transposed
    x_idx, y_idx = idx.nonzero()

    is_not_at_edge = np.logical_and.reduce((x_idx > peak_size -1,
                                            x_idx < width-peak_size - 1,
                                            y_idx > peak_size - 1,
                                            y_idx < height - peak_size - 1))

    y_idx = y_idx[is_not_at_edge]
    x_idx = x_idx[is_not_at_edge]

    if len(y_idx) > 1:
        c = peak_size // 2
        # Create an image with only peaks
        peak_image = np.zeros(t_image.shape)
        peak_image[x_idx, y_idx] = t_image[x_idx, y_idx]
        for x, y in zip(y_idx, x_idx):
            roi = peak_image[y-c:y+c+2, x-c:x+c+2]
            max_peak_pos = np.unravel_index(np.argmax(roi), roi.shape)
            max_peak_val = roi[max_peak_pos[0], max_peak_pos[1]]
            peak_image[y-c:y+c+2, x-c:x+c+2] = 0
            peak_image[y-c + max_peak_pos[0], x-c + max_peak_pos[1]] = max_peak_val
        x_idx, y_idx = (peak_image > 0).nonzero()
    return np.stack((x_idx, y_idx), axis=-1)


def localize_centroid(image: np.ndarray, peaks: np.ndarray, dia: int):
    if peaks is None:  # There is no peak found by find_peaks()
        return None
    if dia % 2 != 1:
        raise ValueError('Window diameter only accepts odd integer values.')
    if peaks.size == 0:
        raise ValueError('There are no peaks input')
    # Filter out the peaks too close to the edges
    height, width = image.shape
    x_idx = peaks[:, 0]
    y_idx = peaks[:, 1]
    is_in_range = np.logical_and.reduce((x_idx > 1.5*dia,
                                         x_idx < width - 1.5*dia,
                                         y_idx > 1.5*dia,
                                         y_idx < height - 1.5*dia))
    peaks = peaks[is_in_range, :]

    radius = int((dia + 1)/2)
    x_weight = np.tile(np.arange(1, 2*radius + 1), (2*radius, 1))
    y_weight = x_weight.T
    mask = _create_circular_mask(2*radius, radius=radius)

    peaks_out = np.zeros(peaks.shape)
    for i, row in enumerate(peaks):
        x = row[0]
        y = row[1]
        masked_roi = mask * image[y-radius+1:y+radius+1, x-radius+1:x+radius+1]
        norm = np.sum(masked_roi)
        x_avg = np.sum(masked_roi * x_weight) / norm + (x - radius + 1)
        y_avg = np.sum(masked_roi * y_weight) / norm + (y - radius + 1)
        peaks_out[i, :] = [x_avg, y_avg]
    return peaks_out


def _create_circular_mask(w, center=None, radius: float = None):
    if center is None:
        # Use the middle of the image, which is the average of 1st index 0 and
        # last index w-1
        center = ((w-1)/2, (w-1)/2)
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], w-center[1])

    y_grid, x_grid = np.ogrid[:w, :w]
    # Use broadcasting to calculate the distaces of each element
    dist_from_center = np.sqrt((x_grid - center[0])**2 + (y_grid-center[1])**2)

    mask = dist_from_center <= radius
    return mask


class Aois():
    def __init__(self, coords: np.ndarray, frame: int, frame_avg: int = 1, width: int = 5):
        self._frame_avg = frame_avg
        self._frame = frame
        self.width = width
        self._coords = coords

    def get_all_x(self):
        return self._coords[:, 0]

    def get_all_y(self):
        return self._coords[:, 1]

    @property
    def frame(self):
        return self._frame

    @property
    def frame_avg(self):
        return self._frame_avg

    def __len__(self):
        return self._coords.shape[0]

    def __iter__(self):
        return map(tuple, self._coords)

    def __contains__(self, item):
        if len(item) == 2:
            in_coord = np.array(item)
            return (self._coords == in_coord).all(axis=1).any()
        return False

    def remove_close_aois(self, distance: int = 0):
        x = self.get_all_x()[np.newaxis]
        y = self.get_all_y()[np.newaxis]
        x_diff_squared = (x - x.T)**2
        y_diff_squared = (y - y.T)**2
        dist_arr = np.sqrt(x_diff_squared + y_diff_squared)
        is_diag = np.identity(len(self), dtype=bool)  # Ignore same aoi distance == 0
        is_not_close = np.logical_or(dist_arr > distance, is_diag).all(axis=1)
        new_aois = Aois(self._coords[is_not_close, :],
                        frame=self.frame,
                        frame_avg=self.frame_avg,
                        width=self.width)
        return new_aois

    def is_in_range_of(self, ref_aois: 'Aois', radius: Union[int, float]):
        x = self.get_all_x()[:, np.newaxis]
        y = self.get_all_y()[:, np.newaxis]
        ref_x = ref_aois.get_all_x()[np.newaxis]  # Produce row vector
        ref_y = ref_aois.get_all_y()[np.newaxis]
        dist_arr = np.sqrt((x - ref_x)**2 + (y - ref_y)**2)
        return (dist_arr <= radius).any(axis=1)

    def remove_aois_near_ref(self, ref_aois: 'Aois', radius: Union[int, float]) -> 'Aois':
        is_in_range_of_ref = self.is_in_range_of(ref_aois=ref_aois, radius=radius)
        new_aois = Aois(self._coords[np.logical_not(is_in_range_of_ref), :],
                        frame=self.frame,
                        frame_avg=self.frame_avg,
                        width=self.width)
        return new_aois

    def remove_aois_far_from_ref(self, ref_aois: 'Aois', radius: Union[int, float]) -> 'Aois':
        is_in_range_of_ref = self.is_in_range_of(ref_aois=ref_aois, radius=radius)
        new_aois = Aois(self._coords[is_in_range_of_ref, :],
                        frame=self.frame,
                        frame_avg=self.frame_avg,
                        width=self.width)
        return new_aois
