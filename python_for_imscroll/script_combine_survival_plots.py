from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
COLORS = ["#a09b00",
          "#6572fe",
          "#d30055"]
FILL_COLORS = ['#e2e1b2',
               '#d0d4fe',
               '#f1b2cc']

def main():
    datapath = Path('/run/media/tzu-yu/data/PriA_project/Analysis_Results/20200228/20200228imscroll/')
    filestr = ['L1', 'L2', 'L3']
    labels = ['500 pM', '125 pM', '62.5 pM']
    fig, ax = plt.subplots()

    for file, label, color, fill_color in zip(filestr, labels, COLORS, FILL_COLORS):
        filepath = datapath / '{}__first_dwellr.hdf5'.format(file)
        with h5py.File(filepath, 'r') as f:
            surv_data = f.get('/survival_curve/data')[()]
            k = f.get('/exp_model/k')[()]

        time = surv_data[0]
        surv = surv_data[1]
        x = np.linspace(0, time[-1], time[-1]*10)
        y = np.exp(-k*x)

        ax.step(time, surv, where='post', color=color, label=label)
        ax.plot(x, y, color=color)
        ax.fill_between(time, surv_data[2], surv_data[3], step='post', color=fill_color)
    ax.set_ylim((0, 1.05))
    ax.set_xlim((0, time[-1]))
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Survival probability', fontsize=14)
    ax.legend()
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(datapath / 'temp.svg', format='svg',
                dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
