from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from python_for_imscroll import utils


def main():
    def change_width(ax, new_value) :
        for patch in ax.patches :
            current_width = patch.get_width()
            diff = current_width - new_value

            # we change the bar width
            patch.set_width(new_value)

            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)

    filepath = Path('~/Analysis_Results/20200922/0806-0922_nucleotide_compile.xlsx').expanduser()
    dfs = utils.read_excel(filepath)
    print(dfs)
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    sns.barplot(x='nucleotide', y='k_on', data=dfs, ci='sd', capsize=0.08,edgecolor='black', fill=False, linewidth=1.5, errwidth=2)
    change_width(ax, .5)
    sns.stripplot(x='nucleotide', y='k_on', jitter=True, data=dfs,  marker='o', color='w', edgecolors='black', linewidth=1, s=6)
    ax.set_ylabel(r'$k_{obs}$ (s)', fontsize=18)
    ax.set_xlabel('Nucleotide', fontsize=18)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=14)
    sns.despine(fig, ax)
    fig.savefig(filepath.parent / 'temp_nucleotide_obs.svg', format='svg',
                dpi=300, bbox_inches='tight')
    fig.clf()

    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    sns.barplot(x='nucleotide', y='k_off', data=dfs, ci='sd', capsize=0.08,edgecolor='black', fill=False, linewidth=1.5, errwidth=2)
    change_width(ax, .5)
    sns.stripplot(x='nucleotide', y='k_off', jitter=True, data=dfs,  marker='o', color='w', edgecolors='black', linewidth=1, s=6)
    ax.set_ylabel(r'$k_{off}$ (s)', fontsize=18)
    ax.set_xlabel('Nucleotide', fontsize=18)
    ax.tick_params(labelsize=14)
    sns.despine(fig, ax)
    fig.savefig(filepath.parent / 'temp_nucleotide_off.svg', format='svg',
                dpi=300, bbox_inches='tight')
    fig.clf()

if __name__ == '__main__':
    main()
