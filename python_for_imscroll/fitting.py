#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (fitting.py) is part of python_for_imscroll.
#
#  python_for_imscroll is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  python_for_imscroll is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with python_for_imscroll.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
from scipy import optimize


def main(x, y, yerr=None):
    if len(x) != len(y):
        raise ValueError('Unequal x and y length')
    popt, _ = fit_linear(x, y, yerr=yerr)
    r_squared = calculate_linear_r_squared(x, y, popt)
    return {'slope': popt[0],
            'intercept': popt[1],
            'r_squared': r_squared,}

def fit_linear(x, y, yerr=None):
    popt, pcov = optimize.curve_fit(f, x, y, sigma=yerr)
    return popt, pcov


def calculate_linear_r_squared(x, y, popt):
    residuals = y - f(x, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1 - (ss_res / ss_tot)

def f(x, a, b):
    x = np.array(x)
    return a * x + b
