#!/usr/bin/env python3
"""

Add a fake image as test data. The faked image is generated as 400x600
array, with 20 frames. Each frame is consecutive integer from the frame
number (starting from 1) to frame number + 400x600 -1, reshaped row by
row.

"""
import pathlib
import numpy as np
from python_for_imscroll import image_processing as imp

def test_read_glimpse_image():
    image_path = pathlib.Path(__file__).parent / 'test_data/fake_im/'
    image_sequence = imp.ImageSequence(image_path)
    n_pixels = image_sequence.width * image_sequence.height
    for i_frame in range(image_sequence.length):
        image = image_sequence.get_one_frame(i_frame)
        true_image = np.reshape(np.arange(i_frame + 1, i_frame + n_pixels + 1),
                                (image_sequence.width, image_sequence.height))
        np.testing.assert_equal(true_image, image)


def test_image_sequence_class():
    image_path = pathlib.Path(__file__).parent / 'test_data/fake_im/'
    image_sequence = imp.ImageSequence(image_path)
    assert image_sequence.width == 600
    assert image_sequence.height == 400
    assert image_sequence.length == 20
