from pathlib import Path
import scipy.io as sio
import numpy as np
from skimage import exposure
import skimage.io
from python_for_imscroll import binding_kinetics



class ImageSequence:
    def __init__(self, image_path):
        header_file_path = image_path / 'header.mat'

        header_file = sio.loadmat(header_file_path)
        self._offset = header_file['vid']['offset'][0, 0].squeeze()
        self._file_number = header_file['vid']['filenumber'][0, 0].squeeze()
        self.width = header_file['vid']['width'][0, 0].item()
        self.height = header_file['vid']['height'][0, 0].item()
        self.length = header_file['vid']['nframes'][0, 0].item()


def load_image_one_frame(frame, header_file_path):
    header_file = sio.loadmat(header_file_path)
    offset = header_file['vid']['offset'][0, 0].squeeze()
    file_number = header_file['vid']['filenumber'][0, 0].squeeze()
    width = header_file['vid']['width'][0, 0].item()
    height = header_file['vid']['height'][0, 0].item()
    print(width, height)
    n_pixels = width * height
    image_file_path = header_file_path.parent / '{}.glimpse'.format(file_number[frame - 1])
    dt = np.dtype('>i2')
    print(dt.name)

    arr = np.fromfile(image_file_path, dtype='>i2', count=n_pixels, offset=offset[frame - 1])
    print(arr.shape)
    arr2 = np.reshape(arr, (height, width))
    arr2 = np.transpose(arr2)
    print(arr2.shape)
    arr2 += 2**15
    print(arr2[0])
    return arr2
