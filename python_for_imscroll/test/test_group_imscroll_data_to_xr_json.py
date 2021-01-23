#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (test_group_imscroll_data_to_xr_json.py) is part of python_for_imscroll.
#
#  python_for_imscroll is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  python_for_imscroll is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with python_for_imscroll.  If not, see <https://www.gnu.org/licenses/>.

from pathlib import Path
import xarray as xr
from python_for_imscroll import group_imscroll_data_to_xr_json as grp
from python_for_imscroll import imscrollIO


# def test_group_imscroll_data():

#     test_data_path = Path(__file__).parent / 'test_data/20200228/'
#     test_parameter_file_path = test_data_path / '20200228parameterFile.xlsx'
#     data = grp.group_imscroll_data_to_xr_json(test_parameter_file_path, ['L2'],
#                                               datapath=test_data_path,
#                                               save_file=False)
#     true_data = imscrollIO.load_data_from_json(test_data_path / 'L2_data.json')
#     for channel in true_data.channel:
#         channel = str(channel.values)

#         xr.testing.assert_equal(data.sel(channel=channel), true_data.sel(channel=channel))
