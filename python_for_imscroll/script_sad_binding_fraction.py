from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from python_for_imscroll import binding_kinetics, visualization

def main():
    datapath = Path('/run/media/tzu-yu/data/PriA_project/Analysis_Results/20191127/20191127imscroll/')
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
    dfs = pd.DataFrame(list(zip(result['green'], result['red'])),
                       columns=['1 nM GstPriA', '70 nM SaDnaD'])
    fig, ax = plt.subplots(figsize=(4, 4.8))
    sns.pointplot(data=dfs, ci='sd', join=False, capsize=0.05)
    sns.stripplot(data=dfs, ax=ax, color='gray', marker='X')
    ax.set_ylim(bottom=0)
    ax.set_ylabel('Colocalized fraction', fontsize=18)
    ax.set_yticks(np.arange(0, 0.5, 0.1))
    ax.tick_params(labelsize=14)
    fig.savefig(datapath / 'temp_colocalization.svg', format='svg', Transparent=True,
                dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
