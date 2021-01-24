from pathlib import Path
import re
import numpy as np
import pytest
from python_for_imscroll import time_series

TEST_DATA_DIR=Path(__file__).parent / 'test_data'

def test_time_traces():
    traces = time_series.TimeTraces.from_xarray_json_all_file(TEST_DATA_DIR / '20200228/L2_all.json')
    n_frames = traces.get_n_traces()
    assert isinstance(n_frames, int)

    traces = time_series.TimeTraces(channels={'blue': np.arange(10), 'green': np.arange(20)+0.5}, n_traces=10)
    assert traces.n_traces == 10

    channels = traces.get_channels()
    assert isinstance(channels, list)
    for i in channels:
        assert isinstance(i, str)
    assert channels == ['blue', 'green']


    expected_exception_str = 'Input array length (11) does not match n_traces (10).'
    with pytest.raises(ValueError, match=re.escape(expected_exception_str)) as exception:
        traces.set_value('intensity', channel='blue', time=0, array=np.arange(11))

    fake_data = np.arange(10)
    for i in range(10):
        traces.set_value('intensity', channel='blue', time=0+i, array=fake_data+i*2)
    correct_data = np.arange(0, 20, 2)
    for i in range(10):
        intensity = traces.get_intensity('blue', i)
        np.testing.assert_equal(intensity, correct_data+i)

    traces.set_value('is_colocalized', channel='blue', time=0+i, array=fake_data+i*2)
