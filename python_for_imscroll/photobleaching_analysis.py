#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (photobleaching_analysis.py) is part of python_for_imscroll.
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
import numpy as np
import pandas as pd
from python_for_imscroll import imscrollIO
from python_for_imscroll import binding_kinetics
from python_for_imscroll import utils


def exclude_inintial_no_binding_aois(intervals):
    not_is_first_interval_0 = intervals.state_number.isel(interval_number=0) != 0
    selected_aois = intervals.AOI[not_is_first_interval_0]
    selected_intevals = intervals.sel(AOI=selected_aois.values)
    return selected_intevals



def main():
    # xlspath = Path('D:/TYL/PriA_project/Analysis_Results/20190917/20190917parameterFile.xlsx')
    xlspath = Path('/mnt/linuxData/Research/PriA_project/analysis_result/20191002/20191002parameterFile.xlsx')

    datapath = imscrollIO.def_data_path()
    # datapath = Path('/mnt/linuxData/Research/PriA_project/analysis_result/20190911/20190911imscroll')
    sheet_list = ['L7']
    specified_n_state = 1
    state_list = ['low', 'high']
    on_off_str = ['on', 'off']
    obs_off_str = ['obs', 'off']
    im_format = 'svg'
    for i_sheet in sheet_list:
        dfs = utils.read_excel(xlspath, sheet_name=i_sheet)
        nFiles = dfs.shape[0]
        interval_list = []
        total_number = 0
        total_cat = {}
        for iFile in range(0, nFiles):
            filestr = dfs.filename[iFile]
            try:
                all_data, AOI_categories = binding_kinetics.load_all_data(datapath / (filestr + '_all.json'))
            except FileNotFoundError:
                print('{} file not found'.format(filestr))
                continue

            print(filestr + ' loaded')

            AOIs_with_binding_event = []
            for key in AOI_categories['analyzable'].keys():
                if key != '0':
                    AOIs_with_binding_event += AOI_categories['analyzable'][key]

            selected_intervals = all_data['intervals'].sel(AOI=AOIs_with_binding_event)
            selected_intervals2 = exclude_inintial_no_binding_aois(selected_intervals)
            number_of_good_traces = 0
            categorizing_good_traces = {}

            for aoi in selected_intervals2.AOI:
                aoi_interval = selected_intervals2.sel(AOI=aoi)
                num_states = int(max(aoi_interval.state_number))
                ar = np.arange(num_states, -1, -1)
                if np.array_equal(aoi_interval.state_number.dropna(dim='interval_number'), ar):
                    number_of_good_traces += 1
                    total_number += 1
                    if num_states not in categorizing_good_traces:
                        categorizing_good_traces[num_states] = 1
                    else:
                        categorizing_good_traces[num_states] += 1
                    if num_states not in total_cat:
                        total_cat[num_states] = 1
                    else:
                        total_cat[num_states] += 1
            print(number_of_good_traces)
            print(categorizing_good_traces)
            print(filestr + ' finished')

        print(total_number)
        print(total_cat)


if __name__ == '__main__':
    main()
