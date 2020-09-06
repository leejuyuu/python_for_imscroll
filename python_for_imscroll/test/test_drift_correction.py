
from pathlib import Path
import numpy as np
import scipy.io as sio
import python_for_imscroll.drift_correction as dcorr

TEST_DATA_DIR = Path(__file__).parent / 'test_data'


def test_make_driftlist_simple():
    driftfit = sio.loadmat(TEST_DATA_DIR / '20200228/L2_driftfit.dat')['aoifits']
    driftlist = dcorr.make_drift_list_simple(driftfit)
    correct_driftlist = sio.loadmat(TEST_DATA_DIR / '20200228/L2_driftlist.dat')['driftlist']
    np.testing.assert_allclose(driftlist, correct_driftlist[:, :3], rtol=0, atol=1e-11)
