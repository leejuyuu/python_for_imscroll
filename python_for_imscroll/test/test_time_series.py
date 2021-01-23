from pathlib import Path
from python_for_imscroll import time_series

TEST_DATA_DIR=Path(__file__).parent / 'test_data'

def test_time_traces():
    traces = time_series.TimeTraces.from_xarray_json_all_file(TEST_DATA_DIR / '20200228/L2_all.json')
    n_frames = traces.get_n_traces()
    assert isinstance(n_frames, int)

    channels = traces.get_channels()
    assert isinstance(channels, list)
    for i in channels:
        assert isinstance(i, str)
