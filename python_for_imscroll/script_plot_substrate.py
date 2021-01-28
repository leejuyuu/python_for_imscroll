
from pathlib import Path
import string
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.optimize
import seaborn as sns
from python_for_imscroll import utils



def main():
    plt.style.use(str(Path('./temp_style.mplstyle').resolve()))
    path = Path('/home/tzu-yu/Analysis_Results/20201224/20201224_substrate_colocalization_count_compile.ods')
    df = pd.read_excel(path, engine='odf')
    columns = df.columns.tolist()[1:]  # first column is the concentration
    dates = sorted(list({i[:-2] for i in columns}))
    colocalized_fraction = pd.DataFrame({date: df[date+'-2']/df[date+'-1'] for date in dates})
    alphabets = string.ascii_lowercase[:df.shape[0]]
    x = df.iloc[:, 0].to_numpy()[:, np.newaxis]
    x = np.arange(df.shape[0])[:, np.newaxis]
    x = np.array(list(alphabets))[:, np.newaxis]
    y = np.nanmean(colocalized_fraction, axis=1)
    y_err = np.nanstd(colocalized_fraction, axis=1)
    # langumuir = lambda x, A, Kd: A*x/(Kd+x)

    x_all = np.tile(x, (1, len(dates))).flatten()
    y_all = colocalized_fraction.to_numpy().flatten()
    is_not_nan = np.logical_not(np.isnan(y_all))
    x_all = x_all[is_not_nan]
    y_all = y_all[is_not_nan]

    # ini_A = y_all.max()
    # ini_Kd = x_all[np.argmin(np.abs(y_all - ini_A/2))].item()

    # popt, _ = scipy.optimize.curve_fit(langumuir, x_all, y_all, p0=[ini_A, ini_Kd])
    # print(popt)


    np.random.seed(0)
    fig, ax = plt.subplots(figsize=(2.8, 1.5))

    def change_width(ax, new_value) :
        for patch in ax.patches :
            current_width = patch.get_width()
            diff = current_width - new_value

            # we change the bar width
            patch.set_width(new_value)

            # we recenter the bar
            patch.set_x(patch.get_x() + diff * .5)

    # breakpoint()
    sns.barplot(x=x_all, y=y_all, ci='sd', capsize=0.08,edgecolor='black', fill=False, linewidth=1, errwidth=2)
    change_width(ax, .5)
    sns.stripplot(x=x_all, y=y_all, jitter=True, marker='o', color='w', edgecolors='black', linewidth=1, s=3)
    # ax.errorbar(x, y, yerr=y_err, marker='o', ms=2.5, linestyle='', zorder=2)
    # line_x = np.linspace(x.min(), x.max(), 1000)
    # ax.plot(line_x, langumuir(line_x, *popt), zorder=0)

    x_jitter = 0.05 * np.random.standard_normal((len(df.index), len(dates)))
    # ax.scatter(x=(x + x_jitter).flatten(), y=colocalized_fraction.to_numpy().flatten(),
    #            marker='o', color='w', edgecolors='gray', linewidth=0.5, s=5, zorder=3)
    ax.set_xlabel('Substrate')
    ax.set_ylabel('Colocalized\nDNA fraction')
    ax.set_ylim(bottom=0)
    # ax.text(0.5, 0.4, r'$K_d$ = {:.1f} pM'.format(popt[1]*1000), transform=ax.transAxes)
    save_fig_path = path.parent / 'plot_substrate.svg'
    fig.savefig(save_fig_path, format='svg')






if __name__ == '__main__':
    main()
