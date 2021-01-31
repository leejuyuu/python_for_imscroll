

from pathlib import Path
from typing import Tuple

import rpy2.robjects as robjects
import rpy2.robjects.pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

import numpy as np  # numpy needs to be imported after rpy2
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import lifelines


def main():
    xls_path = Path(input('Enter excel table path: '))
    # xls_path = Path('/home/tzu-yu/censoring.xlsx')
    df = pd.read_excel(xls_path)
    df = df.rename(columns={'censored':'status'})
    save_fig_path = xls_path.with_suffix('.svg')
    call_r_survival(df, save_fig_path)


def call_r_survival(df: pd.DataFrame, save_path: Path):
    rpy2.robjects.pandas2ri.activate()
    survival = importr('survival')

    with localconverter(robjects.default_converter + rpy2.robjects.pandas2ri.converter):
        r_from_pd_df = robjects.conversion.py2rpy(df)
    robjects.r('surv <- with({}, Surv(time=time, time2=time, event=status, type="interval"))'.format(r_from_pd_df.r_repr()))
    robjects.r('fit <- survfit(surv~1, data={})'.format(r_from_pd_df.r_repr()))
    robjects.r('fit0 <- survfit0(fit)')
    time = robjects.r('fit0[["time"]]')
    surv = robjects.r('fit0[["surv"]]')
    upper_ci = robjects.r('fit0[["upper"]]')
    lower_ci = robjects.r('fit0[["lower"]]')

    robjects.r('exreg <- survreg(surv~1, data={}, dist="exponential")'.format(r_from_pd_df.r_repr()))
    intercept = robjects.r('exreg["coefficients"]')[0].item()
    log_var = robjects.r('exreg["var"]')[0].item()

    # upper_bound = df['time'].where(cond=(df['status'] != 0), other=np.inf)
    # lower_bound = df['time'].where(cond=(df['status'] != 2), other=0)
    # breakpoint()
    # kmf = lifelines.KaplanMeierFitter()
    # kmf.fit_interval_censoring(lower_bound, upper_bound)
    # kmf.plot_survival_function()
    # breakpoint()


    k = np.exp(-intercept)
    print('k = {}'.format(k))
    confidence_level = .95
    alpha = 1-(1-confidence_level)/2
    tau_ci = np.exp(intercept + scipy.stats.norm.ppf(alpha)*np.array([1, -1])*log_var)
    x = np.linspace(0, time[-1], int(round(time[-1]*10)))
    y = np.exp(-k*x)


    fig, ax = plt.subplots()
    ax.step(time, surv, where='post')
    ax.plot(x, y)
    ax.fill_between(time, lower_ci, upper_ci, step='post', color='#a1c9f450')

    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_ylim((0, 1.05))
    ax.set_xlim((0, time[-1]))
    ax.set_xlabel('Time (s)', fontsize=14)
    ax.set_ylabel('Survival probability', fontsize=14)

    k_str = r'$k$ = {:.1f} s'.format(1/k.item())
    plt.text(0.6, 0.85, k_str, transform=ax.transAxes, fontsize=14)
    ci_string = 'ci: [{:.1f}, {:.1f}]'.format(tau_ci[0], tau_ci[1])
    plt.text(0.6, 0.75, ci_string, transform=ax.transAxes, fontsize=14)

    plt.savefig(save_path, format='svg', Transparent=True,
                dpi=300, bbox_inches='tight')
    plt.close()
    data_file_path = save_path.with_suffix('.hdf5')
    with h5py.File(data_file_path, 'w') as f:
        column_keys = np.array(['time', 'survival', 'upper_ci', 'lower_ci'], dtype='S')
        group_survival_curve = f.create_group('survival_curve')
        group_survival_curve.create_dataset('column_keys', column_keys.shape,
                                            column_keys.dtype, column_keys)
        surv_data = np.stack([time, surv, upper_ci, lower_ci])
        group_survival_curve.create_dataset('data', surv_data.shape,
                                            'f', surv_data)
        group_exp_model = f.create_group('exp_model')
        group_exp_model.create_dataset('k', (1,), 'f', k)
        group_exp_model.create_dataset('log_variance', (1,), 'f', log_var)

    a = np.append(k.item(), 1/tau_ci)
    np.savetxt(save_path.with_suffix('.txt'), a)



if __name__ == '__main__':
    main()
