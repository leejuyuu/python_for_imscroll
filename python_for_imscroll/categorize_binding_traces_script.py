#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (categorize_binding_traces_script.py) is part of python_for_imscroll.
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

"""Categorize_binding_traces_script """

from typing import List
from pathlib import Path
import pandas as pd
from python_for_imscroll import imscrollIO
from python_for_imscroll import binding_kinetics as bk


def categorize_binding_traces(parameter_file_path: Path, sheet_list: List[str], datapath: Path = None,
                              save_file=True):
    """Categorize traces into bad and analyzable categories.

    Run through each entry in the specified sheets in the parameter file.
    Args:
        parameter_file_path: The path of the xlsx parameter file
        sheet_list: A list of sheet names to be analyzed.
    """
    if datapath is None:
        datapath = imscrollIO.def_data_path()
    for i_sheet in sheet_list:
        dfs = pd.read_excel(parameter_file_path, sheet_name=i_sheet)
        for filestr in dfs.filename:
            is_ctl_file = filestr[-3:] == 'ctl'
            AOI_categories = {}

            try:
                data = imscrollIO.load_data_from_json(datapath / (filestr + '_data.json'))
            except FileNotFoundError:
                continue

            state_info = bk.collect_all_channel_state_info(data)
            if is_ctl_file:
                bad_tethers = bk.list_none_ctl_positions(state_info.sel(channel=data.target_channel))
                AOI_categories['tethers'] = bad_tethers
            else:
                bad_tethers = bk.list_multiple_tethers(state_info.sel(channel=data.target_channel))
                AOI_categories['multiple_tethers'] = bad_tethers
            remaining_aois = list(set(data.AOI.values) - set(bad_tethers))
            selected_data = data.sel(AOI=remaining_aois)
            protein_channel_data = bk.get_channel_data(selected_data, data.binder_channel[0])
            non_colocalized_aois, intervals = bk.colocalization_analysis(protein_channel_data)
            AOI_categories['false_binding'] = non_colocalized_aois
            analyzable_aois = list(set(remaining_aois) - set(non_colocalized_aois))
            good_intervals = intervals.sel(AOI=analyzable_aois)
            AOI_categories['analyzable'] = \
                bk.group_analyzable_aois_into_state_number(good_intervals)
            all_data = {'data': data,
                        'intervals': intervals,
                        'state_info': state_info}
            if save_file:
                bk.save_all_data(all_data, AOI_categories, datapath / (filestr + '_all.json'))

            print(filestr + ' finished')
    return all_data, AOI_categories

def main():
    """main function"""
    xlsx_parameter_file_path = imscrollIO.get_xlsx_parameter_file_path()
    sheet_list = imscrollIO.input_sheets_for_analysis()
    categorize_binding_traces(xlsx_parameter_file_path, sheet_list)


if __name__ == '__main__':
    main()
