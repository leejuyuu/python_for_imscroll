from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    filepath = Path('/run/media/tzu-yu/linuxData/Research/PriA_project/analysis_result/20200312/20200312_compiled_dwell_for_graph.ods')
    dfs = pd.read_excel(filepath, engine='odf')
    ax = sns.stripplot(x='nucleotide', y='tobs', jitter=False, data=dfs)
    ax.set_ylabel(r'$\tau_{obs}$ (s)', fontsize=14)
    ax.set_xlabel('Nucleotide', fontsize=14)
    fig = ax.get_figure()
    fig.savefig(filepath.parent / 'temp_nucleotide_obs.svg', format='svg', Transparent=True,
                dpi=300, bbox_inches='tight')
    fig.clf()

    ax2 = sns.stripplot(x='nucleotide', y='toff', jitter=False, data=dfs)
    ax2.set_ylabel(r'$\tau_{off}$ (s)', fontsize=14)
    ax2.set_xlabel('Nucleotide', fontsize=14)
    fig2 = ax2.get_figure()
    fig2.savefig(filepath.parent / 'temp_nucleotide_off.svg', format='svg', Transparent=True,
                dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
