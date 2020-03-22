from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    xlspath = Path('/run/media/tzu-yu/data/PriA_project/Analysis_Results/20200317/20200317_photobleach_count.ods')
    dfs = pd.read_excel(xlspath, engine='odf')
    print(dfs)
    fig, ax = plt.subplots(figsize=(4, 6))
    # ax = sns.barplot(x='Category', y='Counts', data=dfs, ci=None)
    # dfs.plot(x='Category', y='Counts', kind='bar', ax=ax)

    ax.bar(dfs.Category, dfs.Counts, width=0.4, edgecolor='black', fill=False)
    ax.set_xlim((-0.5, 1.5))
    ax.set_ylim((0, 240))
    ax.set_ylabel('Molecule counts', fontsize=16)
    ax.tick_params(labelsize=12)
    total = sum(dfs.Counts)
    # Add this loop to add the annotations
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.annotate('{:.1%}'.format(height/total),
                    (x + .5*width, y + height + 5),
                    ha='center',
                    fontsize=12)
    datapath = xlspath.parent
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(datapath / 'photobleach.svg', format='svg', Transparent=True,
                dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
