#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import seaborn as sns
import python_for_imscroll.visualization as vis
from python_for_imscroll import fitting
from python_for_imscroll import utils


def main():
    path = Path('~/Analysis_Results/20201022/0806-1022_conc_compile.xlsx').expanduser()
    df = utils.read_excel(path)

    conc = df['x']
    k_obs = df['k_on']
    k_off = df['k_off']
    print(np.mean(k_off))
    x = np.array(list(set(conc)))
    y = np.array([np.mean(k_obs[conc == i]) for i in x])
    y_err = np.array([np.std(k_obs[conc == i], ddof=1) for i in x])
    fit_result = fitting.main(x, y, y_err)
    # fit_result = fitting.main(conc, k_obs)
    vis.plot_error_and_linear_fit(x, y, y_err, fit_result,
                                  Path('/home/tzu-yu/test_obs.svg'),
                                  x_label='[PriA] (pM)',
                                  y_label=r'$k_{obs}$ (s$^{-1}$)',
                                  left_bottom_as_origin=True,
                                  x_raw=conc, y_raw=k_obs)

    y = np.array([np.mean(k_off[conc == i]) for i in x])
    y_err = np.array([np.std(k_off[conc == i], ddof=1) for i in x])
    vis.plot_error(x, y, y_err,
                                  Path('/home/tzu-yu/test_off.svg'),
                                  x_label='[PriA] (pM)',
                                  y_label=r'$k_{off}$ (s$^{-1}$)',
                                  left_bottom_as_origin=True,
                                  x_raw=conc, y_raw=k_off,
                   y_top=0.03)
    print(np.mean(k_off)/fit_result['slope'])


if __name__ == '__main__':
    main()
