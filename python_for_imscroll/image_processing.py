from pathlib import Path
import math
from typing import Union, Tuple
import scipy.io as sio
import scipy.interpolate
import scipy.ndimage
import scipy.optimize
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
        frame = int(frame)
        if frame < 0:
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

    def get_averaged_image(self, start=0, size=1):
        start = int(start)
        size = int(size)
        image = 0
        for frame in range(start, start + size):
            image += self.get_one_frame(frame)
        return image / size



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
        gaussian_kernel = normalize(np.exp(-(x/r_noise/2)**2))
    else:
        gaussian_kernel = 1
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
    x_weight = np.tile(np.arange(2*radius), (2*radius, 1))
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


def pick_spots(image, threshold=50, noise_dia=1, spot_dia=5, frame=0, aoi_width=5, frame_avg=1):
    filtered_image = band_pass(image, r_noise=noise_dia, r_object=spot_dia)
    peaks = find_peaks(filtered_image, threshold=threshold, peak_size=spot_dia)
    peak_centroids = localize_centroid(filtered_image, peaks=peaks, dia=spot_dia)
    return Aois(peak_centroids, frame=frame, width=aoi_width, frame_avg=frame_avg)


class Aois():
    def __init__(self, coords: np.ndarray, frame: int, frame_avg: int = 1, width: int = 5, channel=None):
        self._frame_avg = frame_avg
        self._frame = frame
        self.width = width
        self._coords = coords
        self._channel = channel

    def get_all_x(self):
        return self._coords[:, 0]

    def get_all_y(self):
        return self._coords[:, 1]

    @property
    def channel(self):
        return self._channel

    @property
    def frame(self):
        return self._frame

    @property
    def frame_avg(self):
        return self._frame_avg

    def __len__(self):
        if len(self._coords.shape) == 1:
            return 1
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
                        width=self.width,
                        channel=self.channel)
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
                        width=self.width,
                        channel=self.channel)
        return new_aois

    def remove_aois_far_from_ref(self, ref_aois: 'Aois', radius: Union[int, float]) -> 'Aois':
        is_in_range_of_ref = self.is_in_range_of(ref_aois=ref_aois, radius=radius)
        new_aois = Aois(self._coords[is_in_range_of_ref, :],
                        frame=self.frame,
                        frame_avg=self.frame_avg,
                        width=self.width,
                        channel=self.channel)
        return new_aois

    def _get_params(self):
        return {'frame': self.frame,
                'frame_avg': self.frame_avg,
                'width': self.width}

    def iter_objects(self):
        gen_objects = (Aois(self._coords[i, :], **self._get_params())
                       for i in range(len(self)))
        return gen_objects

    def get_subimage_slice(self, width=None) -> Tuple[slice, slice]:
        if width is  None:
            width = self.width
        if len(self) == 1:
            offset = width/2 - 0.5
            if self.width % 2:  # Round to int (array element center)
                center = np.round(self._coords)
            else:  # Round to 0.5 (array grid lines)
                center = np.round(self._coords - 0.5) + 0.5
            bounds = (center[:, np.newaxis] + np.array([[-offset, offset+1]])).astype(int)
            # Move the negative lower bound to 0 to avoid slicing empty array
            bounds = np.clip(bounds, a_min=0, a_max=None)
            return (slice(*bounds[1]), slice(*bounds[0]))
        raise ValueError('Wrong AOI length')

    def gaussian_refine(self, image):
        fit_result = np.zeros((len(self), 5))

        for i, aoi in enumerate(self.iter_objects()):
            subimg_slice = aoi.get_subimage_slice()
            height, width = [slice_obj.stop - slice_obj.start for slice_obj in subimg_slice]
            subimg_origin = [slice_obj.start for slice_obj in subimg_slice]
            subimg_origin.reverse()
            x = np.arange(width)[np.newaxis, :]
            y = np.arange(height)[np.newaxis, :]
            xy = [x, y]
            z = image[subimg_slice].ravel()
            fit_result[i] = fit_2d_gaussian(xy, z)
            fit_result[i, 1:3] += subimg_origin

        new_aois = Aois(fit_result[:, 1:3],
                        frame=self.frame,
                        frame_avg=self.frame_avg,
                        width=self.width,
                        channel=self.channel)
        return new_aois

    def __add__(self, other):
        if isinstance(other, tuple) and len(other) == 2:
            new_aois = Aois(np.concatenate((self._coords, np.array(other)[np.newaxis]), axis=0),
                            frame=self.frame,
                            frame_avg=self.frame_avg,
                            width=self.width,
                            channel=self.channel)
            return new_aois
        raise TypeError('Aois class addition only accepts tuples with len == 2')

    def remove_aoi_nearest_to_ref(self, ref_aoi: Tuple[float, float]) -> 'Aois':
        ref_aoi = np.array(ref_aoi)[np.newaxis]
        dist = np.sum((self._coords - ref_aoi)**2, axis=1)
        removing_idx = np.argmin(dist)
        new_coords = np.delete(self._coords, removing_idx, axis=0)
        new_aois = Aois(new_coords,
                        frame=self.frame,
                        frame_avg=self.frame_avg,
                        width=self.width,
                        channel=self.channel)
        return new_aois

    def to_npz(self, path):
        params = {'frame': self.frame,
                  'width': self.width,
                  'frame_avg': self.frame_avg}
        names = ['key', 'value']
        formats = ['U10', int]
        dtype = dict(names=names, formats=formats)
        params = np.fromiter(params.items(), dtype=dtype, count=len(params))
        np.savez(path, params=params, coords=self._coords, channel=np.array(self.channel, dtype='U5'))

    def to_imscroll_aoiinfo2(self, path):
        aoiinfo = np.zeros((len(self), 6))
        aoiinfo[:, 0] = self.frame
        aoiinfo[:, 1] = self.frame_avg
        aoiinfo[:, 2] = self.get_all_y() + 1
        aoiinfo[:, 3] = self.get_all_x() + 1
        aoiinfo[:, 4] = self.width
        aoiinfo[:, 5] = np.arange(1, len(self) + 1)
        sio.savemat(path.with_suffix('.dat'), dict(aoiinfo2=aoiinfo))

    @classmethod
    def from_npz(cls, path):
        npz_file = np.load(path, allow_pickle=False)
        channel = str(npz_file['channel'])
        if channel == 'None':
            channel = None
        params_arr = npz_file['params']
        params = {row['key']: row['value'] for row in params_arr}
        coords = npz_file['coords']
        aois = cls(coords, channel=channel, **params)
        return aois

    def get_interp2d_grid(self):
        if len(self) == 1:
            offset = self.width/2 - 0.5
            x_start, y_start = self._coords - offset
            x_end, y_end = self._coords - offset + self.width
            return np.ogrid[y_start:y_end, x_start:x_end]
        raise ValueError('Wrong AOI length')

    def get_intensity(self, image):
        if len(self) == 1:
            grid = self.get_interp2d_grid()
            grid = (arr.squeeze() for arr in grid)
            y_max, x_max = image.shape
            f = scipy.interpolate.interp2d(*np.ogrid[:y_max, :x_max], image, fill_value=np.nan)
            interpolated_image = f(*grid)
            intensity = np.sum(interpolated_image)
            return intensity
        raise ValueError('Wrong AOI length')

    def get_background_intensity(self, image):
        if len(self) == 1:
            sub_im_slice = self.get_subimage_slice(width=2*self.width+9)
            origin = [slice_obj.start for slice_obj in sub_im_slice]
            shape = [slice_obj.stop - slice_obj.start for slice_obj in sub_im_slice]
            mask = np.ones(shape, dtype=bool)
            aoi_slice = self.get_subimage_slice(width=2*self.width-1)
            aoi_ogrid = np.ogrid[aoi_slice]
            mask[aoi_ogrid[0] - origin[0], aoi_ogrid[1] - origin[1]] = 0
            sub_im = image[sub_im_slice]
            background = np.median(sub_im[mask])
            return background
        raise ValueError('Wrong AOI length')



def symmetric_2d_gaussian(xy, A, x0, y0, sigma, h):
    x, y = xy
    y = y.T
    denominator = (x - x0)**2 + (y - y0)**2
    return (A*np.exp(-denominator/(2*sigma**2)) + h).ravel()


def fit_2d_gaussian(xy, z):
    x_c = xy[0].mean()
    y_c = xy[1].mean()
    bounds = (0, [2*z.max(), xy[0].max(), xy[1].max(), 10000, 2*z.max()])
    param_0 = [z.max() - z.mean(), x_c, y_c, min(x_c, y_c)/2, z.mean()]
    popt, _ = scipy.optimize.curve_fit(symmetric_2d_gaussian,
                                       xy,
                                       z,
                                       p0=param_0,
                                       bounds=bounds,
                                       ftol=1e-6,
                                       gtol=1e-6,
                                       xtol=1e-10,
                                       max_nfev=10000)
    return popt
