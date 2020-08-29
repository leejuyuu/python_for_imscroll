#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (group_imscroll_data_to_xr_json.py) is part of python_for_imscroll.
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

"""Group_imscroll_data_to_xr_json is a script for collecting imscroll outputs
for further analysis in python."""

from typing import List
from pathlib import Path
import pandas as pd
from python_for_imscroll import imscrollIO


def group_imscroll_data_to_xr_json(xlspath: Path, sheet_list: List[str],
                                   datapath: Path = None,
                                   save_file=True):
    """Group imscroll output files into a single xarray object.

    Run through each entry in the specified sheets in the parameter file.
    Args:
        xlspath: The path of the xlsx parameter file
        sheet_list: A list of sheet names to be analyzed.
    """
    if datapath is None:
        datapath = imscrollIO.def_data_path()
    for i_sheet in sheet_list:
        dfs = pd.read_excel(xlspath, sheet_name=i_sheet)

        for filestr, frame_start, frame_end in zip(dfs.filename, dfs['framestart'], dfs['frame end']):
            data = imscrollIO.initialize_data_from_intensity_traces(datapath, filestr, (frame_start - 1, frame_end))
            data = imscrollIO.import_interval_results(data)
            try:
                data = imscrollIO.import_viterbi_paths(data)
            except FileNotFoundError:
                print('error in {}'.format(filestr))
                continue

            if save_file:
                imscrollIO.save_data_to_json(datapath / (filestr + '_data.json'), data)
    return data


def main():
    """main function"""
    xlsx_parameter_file_path = imscrollIO.get_xlsx_parameter_file_path()
    sheet_list = imscrollIO.input_sheets_for_analysis()
    group_imscroll_data_to_xr_json(xlsx_parameter_file_path, sheet_list)


if __name__ == '__main__':
    main()
