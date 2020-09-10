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
        assert image.shape == (image_sequence.height, image_sequence.width)
        true_image = np.reshape(np.arange(i_frame + 1, i_frame + n_pixels + 1),
                                (image_sequence.height, image_sequence.width))
        np.testing.assert_equal(true_image, image)

    with pytest.raises(ValueError) as exception_info:
        image_sequence.get_one_frame(-1)
        assert exception_info.value == 'Frame number must be positive integers or 0'

    with pytest.raises(ValueError) as exception_info:
        image_sequence.get_one_frame('')
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
    true_output = sio.loadmat(image_path)['out'][:, 0:2] - 1  # First two columns are x, y coords
    output = imp.localize_centroid(test_image, peaks, 5+2)
    assert isinstance(output, np.ndarray)
    assert output.shape[1] == 2
    np.testing.assert_allclose(output, true_output, atol=1e-13)  # Tolerate rounding error


    # If there is no peak found by find_peaks()
    output = imp.localize_centroid(test_image, None, 5+2)
    assert output is None


def test_pick_spots():
    test_image = np.zeros((200, 300))
    aois = imp.pick_spots(test_image, noise_dia=1, spot_dia=5, threshold=50)
    assert isinstance(aois, imp.Aois)


def test_aois_class():
    aois = imp.Aois(np.tile(np.arange(10), (2, 1)).T, frame=0)
    assert aois.width == 5
    assert aois.frame == 0
    assert aois.frame_avg == 1
    true_arr = np.arange(10)
    np.testing.assert_equal(aois.get_all_x(), true_arr)
    np.testing.assert_equal(aois.get_all_y(), true_arr)


    aois = imp.Aois(np.tile(np.arange(10), (2, 1)).T, frame=0, frame_avg=10, width=6)
    assert aois.width == 6
    assert aois.frame_avg == 10

    # These two attributes are protected by property, should not be set outside
    # init
    with pytest.raises(AttributeError) as exception:
        aois.frame = 20
        assert "can't set attribute" in str(exception.value)
    with pytest.raises(AttributeError):
        aois.frame_avg = 1
        assert "can't set attribute" in str(exception.value)

    assert len(aois) == 10

    # Iterator returns tuples of (x, y)
    for i, item in enumerate(aois):
        assert isinstance(item, tuple)
        x, y = item
        assert x == i
        assert y == i

    a = np.arange(10)
    aois = imp.Aois(np.stack([a, 2*a], axis=-1), frame=1)

    np.testing.assert_equal(aois.get_all_x(), a)
    np.testing.assert_equal(aois.get_all_y(), a*2)

    for i, item in enumerate(aois):
        assert isinstance(item, tuple)
        x, y = item
        assert x == i
        assert y == 2*i

    assert (1, 2) in aois
    assert np.array([1, 2]) in aois
    assert (1, 1) not in aois
    assert (1, 1, 1) not in aois

    old_coords = aois._coords
    new_aois = aois + (105.2, 50)
    np.testing.assert_equal(new_aois._coords[:-1, :], old_coords)
    np.testing.assert_equal(new_aois._coords[-1, :], np.array((105.2, 50)))


def test_remove_close_aois():
    # x spacing
    arr = np.ones((7, 2))
    arr[:, 0] = np.array([1, 3, 4, 6, 8, 9, 11]) * 4
    aois = imp.Aois(arr, 0, frame_avg=10, width=10)
    new_aois = aois.remove_close_aois(5)
    assert isinstance(new_aois, imp.Aois)
    np.testing.assert_equal(new_aois.get_all_x(), aois.get_all_x()[[0, 3, 6]])
    np.testing.assert_equal(new_aois.get_all_y(), aois.get_all_y()[[0, 3, 6]])
    assert new_aois.frame == 0
    assert new_aois.frame_avg == 10
    assert new_aois.width == 10

    # y spacing
    arr = np.ones((7, 2))
    arr[:, 1] = np.array([1, 3, 4, 6, 8, 9, 11]) * 4
    aois = imp.Aois(arr, 0)
    new_aois = aois.remove_close_aois(5)
    np.testing.assert_equal(new_aois.get_all_x(), aois.get_all_x()[[0, 3, 6]])
    np.testing.assert_equal(new_aois.get_all_y(), aois.get_all_y()[[0, 3, 6]])

    # Boundary case
    arr = np.ones((4, 2))
    arr[:, 1] = np.array([1, 3, 4, 8]) * 5
    aois = imp.Aois(arr, 0)
    new_aois = aois.remove_close_aois(5)
    np.testing.assert_equal(new_aois.get_all_x(), aois.get_all_x()[[0, 3]])
    np.testing.assert_equal(new_aois.get_all_y(), aois.get_all_y()[[0, 3]])

    # Not on axis
    arr = np.ones((7, 2))
    arr[:, 0] = np.array([1, 3, 4, 6, 8, 9, 11]) * 4 * np.cos(np.pi/3)
    arr[:, 1] = np.array([1, 3, 4, 6, 8, 9, 11]) * 4 * np.sin(np.pi/3)
    aois = imp.Aois(arr, 0)
    new_aois = aois.remove_close_aois(5)
    np.testing.assert_equal(new_aois.get_all_x(), aois.get_all_x()[[0, 3, 6]])
    np.testing.assert_equal(new_aois.get_all_y(), aois.get_all_y()[[0, 3, 6]])

    # Aggregates
    arr = np.ones((7, 2))
    arr[:, 0] = np.array([1, 1, 2, 6, 11, 13, 15])
    arr[:, 1] = np.array([1, 2, 1, 6, 11, 13, 14])
    aois = imp.Aois(arr, 0)
    new_aois = aois.remove_close_aois(5)
    np.testing.assert_equal(new_aois.get_all_x(), aois.get_all_x()[[3]])
    np.testing.assert_equal(new_aois.get_all_y(), aois.get_all_y()[[3]])


def test_Aois_is_in_range_of():
    arr = np.array([[0, 1],
                    [10, 10],
                    [100, 50],
                    [100, 52],
                    [100, 49],
                    [200, 100],
                    [300, 70],
                    [400, 80],
                    [786, 520],
                    [150, 200]])
    ref = np.array([[5, 1],
                    [100, 55],
                    [10, 15],
                    [302, 68],
                    [403, 84],
                    [790, 516]])
    aois = imp.Aois(arr, 0)
    ref_aois = imp.Aois(ref, 0)
    is_in_range = aois.is_in_range_of(ref_aois=ref_aois, radius=5)
    assert len(is_in_range) == len(aois)
    true_arr = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    np.testing.assert_equal(is_in_range, true_arr)

    # Remove aois near ref
    new_aois = aois.remove_aois_near_ref(ref_aois, radius=5)
    np.testing.assert_equal(new_aois.get_all_x(), arr[np.logical_not(true_arr), 0])

    # Remove aois far from ref
    new_aois = aois.remove_aois_far_from_ref(ref_aois, radius=5)
    np.testing.assert_equal(new_aois.get_all_x(), arr[true_arr, 0])


def test_fit_2d_gaussian():
    np.random.seed(1)
    xy_data = np.ogrid[:5, :5]
    xy_data[1] = xy_data[1].T
    param = [500, 2.5, 1.7, 1.5, 100]
    image = (imp.symmetric_2d_gaussian(xy_data, *param) + 10*np.random.standard_normal(25))
    fitted_param = imp.fit_2d_gaussian(xy_data, image)
    np.testing.assert_allclose(fitted_param, param, rtol=0.03)


def test_Aois_iter_objects():
    aois = imp.Aois(np.tile(np.arange(10), (2, 1)).T, frame=0)
    for i, aoi in enumerate(aois.iter_objects()):
        assert aoi.width == 5
        assert aoi.frame == 0
        assert aoi.frame_avg == 1
        assert len(aoi) == 1
        assert (aoi._coords == i).all()


def test_aoi_get_subimage():
    aois = imp.Aois(np.tile(np.arange(10), (2, 1)).T, frame=0)
    for aoi in aois.iter_objects():
        subimage_idx = aoi.get_subimage_slice()
        assert isinstance(subimage_idx, tuple)
        assert len(subimage_idx) == 2
        y_slice, x_slice = subimage_idx
        assert y_slice.stop - y_slice.start == 5
        assert x_slice.stop - x_slice.start == 5


def test_aoi_gaussian_refine():
    image_path = pathlib.Path(__file__).parent / 'test_data/20200228/hwligroup00774'
    image_sequence = imp.ImageSequence(image_path)
    aoiinfo_path = pathlib.Path(__file__).parent / 'test_data/20200228/L2_aoi.dat'
    aoiinfo = sio.loadmat(aoiinfo_path)['aoiinfo2']
    # The imscroll image was transposed, so there is a reversion of the coords
    aois = imp.Aois(aoiinfo[:, [3, 2]] - 1, aoiinfo[0, 0], aoiinfo[0, 1], aoiinfo[0, 4])
    true_image_path = pathlib.Path(__file__).parent / 'test_data/20200228/averaged_im.mat'
    true_image = sio.loadmat(true_image_path)['currentFrameImage'].T
    new_aois = aois.gaussian_refine(true_image)
    aoifits_path = pathlib.Path(__file__).parent / 'test_data/20200228/10_frame_fit.dat'
    aoifits = sio.loadmat(aoifits_path)['aoifits']['data'][0, 0]
    np.testing.assert_allclose(new_aois._coords, aoifits[:, [4, 3]] - 1, atol=0.0003, rtol=0)

    # Tolerate more error since I will use float averaged_image here (not rounded)
    averaged_image = image_sequence.get_averaged_image(start=aoiinfo[0, 0]-1, size=aoiinfo[0, 1])
    new_aois = aois.gaussian_refine(averaged_image)
    np.testing.assert_allclose(new_aois._coords, aoifits[:, [4, 3]] - 1, atol=0.0015, rtol=0)


def test_get_averaged_image():
    # Check that the averaged image is withing rounding error
    image_path = pathlib.Path(__file__).parent / 'test_data/20200228/hwligroup00774'
    image_sequence = imp.ImageSequence(image_path)
    averaged_image = image_sequence.get_averaged_image(start=0, size=10)
    true_path = pathlib.Path(__file__).parent / 'test_data/20200228/averaged_im.mat'
    true_image = sio.loadmat(true_path)['currentFrameImage'].T
    np.testing.assert_allclose(averaged_image, true_image, atol=0.5)

def test_remove_aoi_nearest_to_ref():
    arr = np.array([[0, 1],
                    [10, 10],
                    [100, 50],
                    [100, 52],
                    [100, 49],
                    [200, 100],
                    [300, 70],
                    [400, 80],
                    [786, 520],
                    [150, 200]])
    aois = imp.Aois(arr, 5, 10, 7)
    new_aois = aois.remove_aoi_nearest_to_ref((0.5, 2))
    assert isinstance(new_aois, imp.Aois)
    assert new_aois.width == aois.width
    assert new_aois.frame == aois.frame
    assert new_aois.frame_avg == aois.frame_avg
    assert new_aois.channel == aois.channel
    np.testing.assert_equal(new_aois._coords, arr[1:, :])

def test_save_load_aois():
    np.random.seed(0)
    arr = np.random.random_sample((20, 2)) * 512
    aois = imp.Aois(arr, 5, 10, 7)
    save_path = pathlib.Path('/run/media/tzu-yu/linuxData/Git_repos/python_for_imscroll/python_for_imscroll/test/test_data/save_path/aois')
    actual_path = save_path.with_suffix('.npz')
    if actual_path.exists() and actual_path.is_file():
        actual_path.unlink()
    aois.to_npz(save_path)
    assert actual_path.exists()
    assert actual_path.is_file()

    new_aois = imp.Aois.from_npz(actual_path)
    assert isinstance(new_aois, imp.Aois)
    assert new_aois.width == aois.width
    assert new_aois.frame == aois.frame
    assert new_aois.frame_avg == aois.frame_avg
    assert new_aois.channel == aois.channel
    np.testing.assert_equal(new_aois._coords, arr)


def test_get_aoi_intensity():
    # Should calculate the correct intensity when the center move gradually out
    # of from the original spot.
    arr = np.full((5, 2), 3, dtype='f')
    arr[:, 0] += np.arange(0, 1, 0.2)
    aois = imp.Aois(arr, 0)

    image = np.zeros((7, 7))
    image[1:-1, 1:-1] = 1
    for i, aoi in enumerate(aois.iter_objects()):
        intensity = aoi.get_intensity(image)
        np.testing.assert_allclose(intensity, 25 - i)

    # Should be fine at all four corners
    arr = np.array([[2, 2],
                    [4, 2],
                    [4, 4],
                    [2, 4]])
    aois = imp.Aois(arr, 0)
    for i, aoi in enumerate(aois.iter_objects()):
        intensity = aoi.get_intensity(image)
        np.testing.assert_allclose(intensity, 16)

    # Should return NaN if the AOI boundary get outside the image boundary
    arr = np.array([[2-0.0001, 2],
                    [4+0.0001, 2],
                    [4, 2-0.0001],
                    [4, 4+0.0001],
                    [2, 4+0.0001]])
    aois = imp.Aois(arr, 0)
    for i, aoi in enumerate(aois.iter_objects()):
        intensity = aoi.get_intensity(image)
        assert np.isnan(intensity)
