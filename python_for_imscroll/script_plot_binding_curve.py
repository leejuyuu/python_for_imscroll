
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.optimize
import seaborn as sns
from python_for_imscroll import utils


# def langumuir(x, A, Kd):
#     retur

def main():
    path = Path('/run/media/tzu-yu/data/PriA_project/Analysis_Results/20201015/20201015_colocalization_count_compile.ods')
    df = pd.read_excel(path, engine='odf')
    columns = df.columns.tolist()[1:]  # first column is the concentration
    dates = sorted(list(set([i[:-2] for i in columns])))
    colocalized_fraction = pd.DataFrame({date: df[date+'-2']/df[date+'-1'] for date in dates})
    x = df.iloc[:, 0].to_numpy()[:, np.newaxis] / 1000
    y = np.nanmean(colocalized_fraction, axis=1)
    y_err = np.nanstd(colocalized_fraction, axis=1)
    langumuir = lambda x, A, Kd: A*x/(Kd+x)

    ini_A = y.max()
    ini_Kd = x[np.argmin(np.abs(y - ini_A/2))].item()
    # interp_f = scipy.interpolate.interp1d(x.squeeze(), y)
    # ini_Kd = scipy.optimize.newton(interp_f, 0.5)

    popt, _ = scipy.optimize.curve_fit(langumuir, x.squeeze(), y, p0=[ini_A, ini_Kd], sigma=y_err)
    print(popt)


    sns.set_palette(palette='muted')
    np.random.seed(0)
    fig, ax = plt.subplots(figsize=(4, 3))

    sns.despine(fig, ax)
    line_x = np.linspace(x.min(), x.max(), 1000)
    ax.plot(line_x, langumuir(line_x, *popt))
    ax.errorbar(x, y, yerr=y_err, marker='o', ms=5, linestyle='', capsize=2)

    x_jitter = 0.05 * np.random.standard_normal((len(df.index), len(dates)))
    ax.scatter(x=(x + x_jitter).flatten(), y=colocalized_fraction.to_numpy().flatten(), marker='o', color='w', edgecolors='gray', linewidth=1, s=14, zorder=3)
    save_fig_path = path.parent / 'plot.svg'
    fig.savefig(save_fig_path, format='svg')






if __name__ == '__main__':
    main()
