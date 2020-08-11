#!/usr/bin/env python3

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def main():
    path = Path('/run/media/tzu-yu/data/PriA_project/Expt_data/20200805_sort_peptide_Dy649_hplc/2020_08_06_N21-1_0_N21_1.txt')
    save_fig_path = path.parent / 'N21.svg'
    arr = np.loadtxt(path, delimiter=' ')
    fig, ax = plt.subplots(figsize=(6, 4))

    mz = arr[:, 0]
    intensity = arr[:, 1]
    max_idx = np.argmax(intensity)
    sns.despine()

    ax.plot(arr[:, 0], arr[:, 1], linewidth=0.5)
    # ax.set_xlim((3050, 3120))
    ax.set_xlim((4080, 4160))
    ax.annotate(f'{mz[max_idx]}', xy=(mz[max_idx], intensity[max_idx]*1.01), ha='center', fontsize=6)

    ax.set_xlabel('m/z', fontsize=16)
    ax.set_ylabel('Intensity (a.u.)', fontsize=16)
    # plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(save_fig_path, format='svg', Transparent=True, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
