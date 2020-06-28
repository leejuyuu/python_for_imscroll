#!/usr/bin/env python3
"""

Add a fake image as test data. The faked image is generated as 400x600
array, with 20 frames. Each frame is consecutive integer from the frame
number (starting from 1) to frame number + 400x600 -1, reshaped row by
row.

"""
import pathlib
import numpy as np
import pytest
import scipy.io as sio
from python_for_imscroll import image_processing as imp

def test_read_glimpse_image():
    image_path = pathlib.Path(__file__).parent / 'test_data/fake_im/'
    image_sequence = imp.ImageSequence(image_path)
    n_pixels = image_sequence.width * image_sequence.height
    for i_frame in range(image_sequence.length):
        image = image_sequence.get_one_frame(i_frame)
        true_image = np.reshape(np.arange(i_frame + 1, i_frame + n_pixels + 1),
                                (image_sequence.height, image_sequence.width))
        np.testing.assert_equal(true_image, image)

    with pytest.raises(ValueError) as exception_info:
        image_sequence.get_one_frame(-1)
        assert exception_info.value == 'Frame number must be positive integers or 0'

    with pytest.raises(ValueError) as exception_info:
        image_sequence.get_one_frame(1.1)
        assert exception_info.value == 'Frame number must be positive integers or 0'

    with pytest.raises(ValueError) as exception_info:
        image_sequence.get_one_frame(image_sequence.length)
        assert (exception_info.value
                == 'Frame number ({}) exceeds sequence length - 1 ({})'.format(image_sequence.length,
                                                                               image_sequence.length - 1))

def test_image_sequence_class():
    image_path = pathlib.Path(__file__).parent / 'test_data/fake_im/'
    image_sequence = imp.ImageSequence(image_path)
    assert image_sequence.width == 300
    assert image_sequence.height == 200
    assert image_sequence.length == 20


def test_iter_over_image_sequece():
    image_path = pathlib.Path(__file__).parent / 'test_data/fake_im/'
    image_sequence = imp.ImageSequence(image_path)
    n_pixels = image_sequence.width * image_sequence.height
    for i_frame, i_frame_image in enumerate(image_sequence):
        true_image = np.reshape(np.arange(i_frame + 1, i_frame + n_pixels + 1),
                                (image_sequence.height, image_sequence.width))
        np.testing.assert_equal(true_image, i_frame_image)


def test_band_pass():
    image_path = pathlib.Path('/run/media/tzu-yu/linuxData/Git_repos/Imscroll/imscroll/test/test_data/test_image.mat')
    test_image = sio.loadmat(image_path)['testImage']
    image_path = pathlib.Path('/run/media/tzu-yu/linuxData/Git_repos/Imscroll/imscroll/test/test_data/test_bpass.mat')
    true_image = sio.loadmat(image_path)['filteredImage']
    filtered_image = imp.band_pass(test_image, 1, 5)
    np.testing.assert_allclose(true_image, filtered_image, atol=1e-12)


def test_band_pass_real_image():
    image_path = pathlib.Path('/run/media/tzu-yu/linuxData/Git_repos/Imscroll/imscroll/test/test_data/test_bpass_real_image.mat')
    test_image = sio.loadmat(image_path)['image']
    true_image = sio.loadmat(image_path)['filteredImage']
    filtered_image = imp.band_pass(test_image, 1, 5)
    np.testing.assert_allclose(true_image, filtered_image, atol=1e-12)


def test_find_peak():
    image_path = pathlib.Path('/run/media/tzu-yu/linuxData/Git_repos/Imscroll/imscroll/test/test_data/test_pkfnd_71_5.mat')
    test_image = sio.loadmat(image_path)['filteredImage']
    true_peaks = sio.loadmat(image_path)['spotCoords']
    peaks = imp.find_peaks(test_image, threshold=71, peak_size=5)
    assert isinstance(peaks, np.ndarray)
    assert peaks.shape[1] == 2
    assert np.issubdtype(peaks.dtype, np.integer)  # Check that the returned type is some integer
    peaks += 1  # Convert to 1 based indexing
    np.testing.assert_equal(peaks, true_peaks)

    # Case2 with different param
    image_path = pathlib.Path('/run/media/tzu-yu/linuxData/Git_repos/Imscroll/imscroll/test/test_data/test_pkfnd_1_5.mat')
    test_image = sio.loadmat(image_path)['filteredImage']
    true_peaks = sio.loadmat(image_path)['spotCoords']
    peaks = imp.find_peaks(test_image, threshold=1, peak_size=5)
    assert isinstance(peaks, np.ndarray)
    assert peaks.shape[1] == 2
    assert np.issubdtype(peaks.dtype, np.integer)  # Check that the returned type is some integer
    peaks += 1  # Convert to 1 based indexing
    np.testing.assert_equal(peaks, true_peaks)


def test_localize_centroid():
    image_path = pathlib.Path('/run/media/tzu-yu/linuxData/Git_repos/Imscroll/imscroll/test/test_data/test_cntrd_71_5.mat')
    test_image = sio.loadmat(image_path)['filteredImage']
    peaks = sio.loadmat(image_path)['spotCoords'] - 1  # Minus 1 to convert to 0 based index
    true_output = sio.loadmat(image_path)['out'][:, 0:2]  # First two columns are x, y coords
    output = imp.localize_centroid(test_image, peaks, 5+2)
    assert isinstance(output, np.ndarray)
    assert output.shape[1] == 2
    np.testing.assert_allclose(output, true_output, atol=1e-13)  # Tolerate rounding error


    # If there is no peak found by find_peaks()
    output = imp.localize_centroid(test_image, None, 5+2)
    assert output is None
