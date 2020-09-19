
from pathlib import Path
import numpy as np
import scipy.io as sio
import python_for_imscroll.drift_correction as dcorr
import python_for_imscroll.image_processing as imp

TEST_DATA_DIR = Path(__file__).parent / 'test_data'


def test_make_driftlist_simple():
    driftfit = sio.loadmat(TEST_DATA_DIR / '20200228/L2_driftfit.dat')['aoifits']
    driftlist = dcorr.make_drift_list_simple(driftfit)
    driftlist[:, 0] += 1
    correct_driftlist = sio.loadmat(TEST_DATA_DIR / '20200228/L2_driftlist.dat')['driftlist']
    np.testing.assert_allclose(driftlist, correct_driftlist[:, :3], rtol=0, atol=1e-11)

def test_drift_corrector():
    driftlist = np.ones((10, 3))
    driftlist[:, 0] = np.arange(10)
    driftlist[:, 2] = 2
    driftlist[0, :] = 0
    drifter = dcorr.DriftCorrector(driftlist)
    np.testing.assert_equal(drifter._driftlist, driftlist)
    arr = np.random.uniform(0, 100, (100, 2))

    # Starting from the first frame
    aois = imp.Aois(arr, frame=0, frame_avg=1, width=7, channel='blue')
    for frame in range(10):
        new_aois = drifter.shift_aois(aois, frame)
        assert isinstance(new_aois, imp.Aois)
        assert new_aois.frame == frame
        assert new_aois.frame_avg == aois.frame_avg
        assert new_aois.width == aois.width
        assert new_aois.channel == aois.channel
        correct_arr = arr[:, :]
        correct_arr[:, 0] += frame
        correct_arr[:, 1] += frame*2
        np.testing.assert_allclose(new_aois._coords, correct_arr)

    # Starting from other frames
    aois = imp.Aois(arr, frame=2, frame_avg=1, width=7, channel='blue')
    for frame in range(2, 10):
        new_aois = drifter.shift_aois(aois, frame)
        assert isinstance(new_aois, imp.Aois)
        assert new_aois.frame == frame
        assert new_aois.frame_avg == aois.frame_avg
        assert new_aois.width == aois.width
        assert new_aois.channel == aois.channel
        correct_arr = arr[:, :]
        correct_arr[:, 0] += frame-2
        correct_arr[:, 1] += (frame-2)*2
        np.testing.assert_allclose(new_aois._coords, correct_arr)
