from pathlib import Path
import numpy as np
from python_for_imscroll import binding_kinetics, visualization

def main():
    datapath = Path('/run/media/tzu-yu/linuxData/Research/PriA_project/analysis_result/20191127/20191127imscroll/')
    filestr = 'L2_02_01'
    result = {'red': [],
                  'green': []}

    for filestr in ['L2_02_01', 'L2_02_02', 'L2_02_03', 'L2_02_04', 'L2_02_05']:
        savedir = datapath / filestr
        try:
            all_data, AOI_categories = binding_kinetics.load_all_data(datapath
                                                                        / (filestr + '_all.json'))
        except FileNotFoundError:
            print('{} file not found'.format(filestr))
        good_aois = []
        for aois in AOI_categories['analyzable'].values():
            good_aois.extend(aois)
        good_interval_traces = all_data['data'].interval_traces.sel(AOI=good_aois)
        total_num = len(good_aois)
        print(total_num)
        for channel in ['green', 'red']:
            pts = []
            interval_traces = good_interval_traces.sel(channel=channel)
            print(np.any(interval_traces != -2).values.item())
            for frame in range(99, 500, 100):

                pts.append(np.count_nonzero(interval_traces.isel(time=frame)% 2))
            print(pts)
            result[channel].append(np.mean(pts)/total_num)
    print(result)



if __name__ == '__main__':
    main()
