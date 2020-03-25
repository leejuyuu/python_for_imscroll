from pathlib import Path
import matplotlib.pyplot as plt


def main():

    datapath = Path('/run/media/tzu-yu/data/PriA_project/Analysis_Results/20200222/')
    tobs = [149.6, 155.1]
    toff = [260.3, 221.9]
    name = ['−', 'SaDnaD-SNAP']

    plt.rcParams['svg.fonttype'] = 'none'

    fig, ax = plt.subplots()
    ax.bar(name, tobs, width=0.3, color='silver')
    ax.set_ylim((0, 200))
    ax.set_xlim((-0.5, 1.5))
    ax.tick_params(labelsize=14)
    ax.set_ylabel(r'$\tau_{on}$ (s)', fontsize=18)
    fig.savefig(datapath / 'temp_SaDnaD-SNAP_obs.svg', format='svg', Transparent=True,
                dpi=300, bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.bar(name, toff, width=0.3, color='silver')
    ax.set_ylim((0, 300))
    ax.set_xlim((-0.5, 1.5))
    ax.tick_params(labelsize=14)
    ax.set_ylabel(r'$\tau_{off}$ (s)', fontsize=18)
    fig.savefig(datapath / 'temp_SaDnaD-SNAP_off.svg', format='svg', Transparent=True,
                dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
