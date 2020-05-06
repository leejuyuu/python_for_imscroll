from pathlib import Path
import scipy.io as sio
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
