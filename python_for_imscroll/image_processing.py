from pathlib import Path
import math
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
        image = np.reshape(image, (self.width, self.height))
        # The glimpse saved image is U16 integer - 2**15 and saved in I16 format.
        # To recover the U16 integers, we need to add 2**15 back, but we cannot
        # do this directly since the image is read as I16 integer, adding 2**15
        # will cause overflow. Need to cast to larger container (at least 32 bit)
        # first.
        image = image.astype(int) + 2**15
        return image

    def __iter__(self):
        return (self.get_one_frame(frame) for frame in range(self.length))
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
