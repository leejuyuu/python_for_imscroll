
from pathlib import Path
import autograd
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.optimize
from python_for_imscroll import utils



def main():
    plt.style.use(str(Path('./temp_style.mplstyle').resolve()))
    path = Path('/home/tzu-yu/Analysis_Results/20210121/20210121_colocalization_count_compile.ods')
    df = pd.read_excel(path, engine='odf')
    columns = df.columns.tolist()[1:]  # first column is the concentration
    dates = sorted(list({i[:-2] for i in columns}))
    colocalized_fraction = pd.DataFrame({date: df[date+'-2']/df[date+'-1'] for date in dates})
    x = df.iloc[:, 0].to_numpy()[:, np.newaxis] / 1000
    y = np.nanmean(colocalized_fraction, axis=1)
    y_err = np.nanstd(colocalized_fraction, axis=1, ddof=1)
    # langumuir = lambda x, A, Kd: A*x/(Kd+x)

    def langumuir(x, A, Kd):
        return A*x/(Kd+x)

    x_all = np.tile(x, (1, len(dates))).flatten()
    y_all = colocalized_fraction.to_numpy().flatten()
    is_not_nan = np.logical_not(np.isnan(y_all))
    x_all = x_all[is_not_nan]
    y_all = y_all[is_not_nan]

    ini_A = y_all.max()
    ini_Kd = x_all[np.argmin(np.abs(y_all - ini_A/2))].item()

    jac_in = jax.jacrev(langumuir, argnums=[1, 2])
    def jac(x, A, Kd):
        return np.asarray(jac_in(x, A, Kd)).T


    popt, pcov = scipy.optimize.curve_fit(langumuir, x.squeeze(), y, sigma=y_err, p0=[ini_A, ini_Kd], jac=jac)
    print(popt)
    print(np.sqrt(np.diagonal(pcov)))


    np.random.seed(0)
    fig, ax = plt.subplots()

    ax.errorbar(x, y, yerr=y_err, marker='o', ms=2.5, linestyle='', zorder=2)
    line_x = np.linspace(x.min(), x.max(), 1000)
    ax.plot(line_x, langumuir(line_x, *popt), zorder=0)

    x_jitter = 0.05 * np.random.standard_normal((len(df.index), len(dates)))
    ax.scatter(x=(x * np.exp(x_jitter)).flatten(), y=colocalized_fraction.to_numpy().flatten(),
               marker='o', color='w', edgecolors='gray', linewidth=0.5, s=5, zorder=3)
    ax.set_xlabel('PriA concentration (nM)')
    ax.set_ylabel('Colocalized\nDNA fraction')
    ax.set_xscale('log')
    ax.set_ylim(bottom=0)
    ax.text(0.5, 0.2, r'$K_d$ = {:.1f} pM'.format(popt[1]*1000), transform=ax.transAxes)
    save_fig_path = path.parent / 'plot.svg'
    fig.savefig(save_fig_path, format='svg')


if __name__ == '__main__':
    main()
