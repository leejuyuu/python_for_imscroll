from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
COLORS = ["#a09b00",
          "#6572fe",
          "#d30055"]
COLORS =["#9671c3",
"#69a75f",
"#cc5366",
"#be883d"]
FILL_COLORS = ['#e2e1b2',
               '#d0d4fe',
               '#f1b2cc',
'']

def main():
    datapath = Path('/run/media/tzu-yu/data/PriA_project/Analysis_Results/20200228/20200228imscroll/')
    filestr = ['L1', 'L2', 'L3', 'L5']
    num = [500, 125, 62.5, 250]
    sorting = np.argsort(num)
    labels = ['500 pM', '125 pM', '62.5 pM', '250 pM']
    filestr = [filestr[i] for i in sorting]
    labels = [labels[i] for i in sorting]
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.despine(fig, ax)

    for file, label, color, fill_color  in zip(filestr, labels, sns.color_palette(palette='muted'), sns.color_palette(palette='pastel')):
        filepath = datapath / '{}__first_dwellr.hdf5'.format(file)
        with h5py.File(filepath, 'r') as f:
            surv_data = f.get('/survival_curve/data')[()]
            param = f.get('/bi_exp_model/param')[()]

        time = surv_data[0]
        surv = surv_data[1]
        S = lambda t, k1, k2, A: A*np.exp(-k1*t) + (1-A)*np.exp(-k2*t)
        x = np.linspace(0, time[-1], int(round(time[-1]*10)))
        y = S(x, *param)

        ax.step(time, surv, where='post', color=color, label=label)
        ax.plot(x, y, color=color)
        fill_color = fill_color + (80/255,)
        ax.fill_between(time, surv_data[2], surv_data[3], step='post', color=fill_color)
    ax.set_ylim((0, 1.05))
    ax.set_xlim((0, time[-1]))
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Survival probability', fontsize=14)
    ax.legend(frameon=False)
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(datapath / 'temp.svg', format='svg', Transparent=True,
                dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
