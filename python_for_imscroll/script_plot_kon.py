#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import python_for_imscroll.visualization as vis
from python_for_imscroll import fitting


def main():
    conc = 62.5*np.array([1, 2, 4, 8, 8, 2, 1, 4])
    time = np.array([185.17893097,
                     123.37271731,
                     35.37462628,
                     36.94108841,
                     56.48242756,
                     212.71951509,
                     585.64238082,
                     156.66647635])
    k_obs = 1/time
    x = np.array(list(set(conc)))
    y = np.array([np.mean(k_obs[conc == i]) for i in x])
    y_err = np.array([np.std(k_obs[conc == i]) for i in x])
    fit_result = fitting.main(x, y, y_err)
    vis.plot_error_and_linear_fit(x, y, y_err, fit_result,
                                  Path('/home/tzu-yu/test.svg'),
                                  x_label='[PriA] (pM)',
                                  y_label=r'$k_{obs}$ (s$^{-1}$)',
                                  left_bottom_as_origin=True)


if __name__ == '__main__':
    main()
