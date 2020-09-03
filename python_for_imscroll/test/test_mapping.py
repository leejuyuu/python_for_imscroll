from pathlib import Path
import numpy as np
import scipy.io as sio
import pytest
from python_for_imscroll import mapping
from python_for_imscroll import image_processing as imp

TEST_DATA_DIR = Path(__file__).parent / 'test_data'

def test_mapper_from_imscroll():
    path = TEST_DATA_DIR / 'mapping/20200206_br_02.dat'
    mapper = mapping.Mapper.from_imscroll(path)
    assert isinstance(mapper, mapping.Mapper)
    matrix = mapper.map_matrix[('blue', 'red')]
    np.testing.assert_equal(matrix, sio.loadmat(path)['fitparmvector'])

    arr = sio.loadmat(TEST_DATA_DIR / 'mapping/L2_aoi.dat')['aoiinfo2']
    aois = imp.Aois(arr[:, 2:4],
                    frame=20,
                    width=7,
                    frame_avg=5,
                    channel='blue')
    for channel in ('red', 'blue'):
        new_aois = mapper.map(aois, to_channel=channel)
        assert isinstance(new_aois, imp.Aois)
        assert new_aois.frame == aois.frame
        assert new_aois.frame_avg == aois.frame_avg
        assert new_aois.width == aois.width
        assert new_aois.channel == channel
        if channel == 'red':
            correct_arr = sio.loadmat(TEST_DATA_DIR / 'mapping/L2_map.dat')['aoiinfo2']
            np.testing.assert_allclose(new_aois._coords, correct_arr[:, 2:4], rtol=1e-14)

    with pytest.raises(ValueError) as exception_info:
        new_aois = mapper.map(aois, to_channel='green')
        assert exception_info.value == ('Mapping matrix from channel blue'
                                        ' to channel green is not loaded')

    for to_channel in ('black', 123):
        with pytest.raises(ValueError) as exception_info:
            new_aois = mapper.map(aois, to_channel=to_channel)
            assert exception_info.value == 'To-channel is not one of the available channels'

    for from_channel in ('black', 123):
        aois = imp.Aois(arr[:, 2:4],
                        frame=20,
                        width=7,
                        frame_avg=5,
                        channel=from_channel)
        with pytest.raises(ValueError) as exception_info:
            new_aois = mapper.map(aois, to_channel=to_channel)
            assert exception_info.value == 'From-channel is not one of the available channels'

    from_channel = 'red'
    channel = 'blue'
    aois = imp.Aois(arr[:, 2:4],
                    frame=20,
                    width=7,
                    frame_avg=5,
                    channel=from_channel)

    new_aois = mapper.map(aois, to_channel=channel)
    assert isinstance(new_aois, imp.Aois)
    assert new_aois.frame == aois.frame
    assert new_aois.frame_avg == aois.frame_avg
    assert new_aois.width == aois.width
    assert new_aois.channel == channel
    correct_arr = sio.loadmat(TEST_DATA_DIR / 'mapping/L2_inv_map.dat')['aoiinfo2']
    np.testing.assert_allclose(new_aois._coords, correct_arr[:, 2:4], rtol=1e-15)
