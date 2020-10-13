#  Copyright (C) 2020 Tzu-Yu Lee, National Taiwan University
#
#  This file (visualization.py) is part of python_for_imscroll.
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

"""Visualization module handles single-molecule fluorescence data visualization"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
import seaborn as sns
from python_for_imscroll import imscrollIO
from python_for_imscroll import binding_kinetics
from python_for_imscroll import utils


def plot_one_trace_and_save(molecule_data: xr.Dataset, category: str = '',
                            save_dir: Path = Path(), save_format: str = 'svg',
                            time_offset: float = 0):
    """Take dataset of one molecule and plot time trajectory.

    Each subplot corresponds to one channel (color).
    Input:
        molecule_data: Dataset of a certain molecule, should contain time and
        channel coordinates, with 'intensity' and 'viterbi_path' variables.

        category: The category of that trace.
        save_dir: The directory to save the image.
        save_format: The image file format to save.
    """
    molecule_number = int(molecule_data.AOI)
    fig = plt.figure(figsize=(10, 5))
    plt.suptitle('molecule {} ({})'.format(molecule_number, category), fontsize=14)
    channel_list = list(set(molecule_data.channel.values.tolist()))
    channel_list.sort()
    for i, i_channel in enumerate(channel_list, 1):
        plt.subplot(len(channel_list), 1, i)
        intensity = molecule_data['intensity'].sel(channel=i_channel)
        vit = molecule_data['viterbi_path'].sel(channel=i_channel, state='position')
        plt.plot(intensity.time + time_offset, intensity, color=i_channel)
        plt.plot(vit.time + time_offset, vit, color='black', linewidth=2)
        if plt.ylim()[0] > 0:
            plt.ylim(bottom=0)
        plt.ylim(plt.ylim())
        if time_offset:
            ax = plt.gca()
            plt.fill([0, time_offset, time_offset, 0],
                    [plt.ylim()[0], plt.ylim()[0], plt.ylim()[1], plt.ylim()[1]],
                    '#e4e4e4')
        plt.xlim((0, molecule_data.time.max() + time_offset))

    fig.text(0.04, 0.4, 'Intensity', ha='center', fontsize=16, rotation='vertical')
    plt.xlabel('time (s)', fontsize=16)
    plt.rcParams['svg.fonttype'] = 'none'
    if time_offset:
        plt.savefig(save_dir / ('molecule{}_shifted.{}'.format(molecule_number, save_format)),
                    dpi=300, bbox_inches='tight', format=save_format)
    else:
        plt.savefig(save_dir / ('molecule{}.{}'.format(molecule_number, save_format)),
                    dpi=300, bbox_inches='tight', format=save_format)
    plt.close()


def main():
    """Plot traces from every molecule in the dataset and save them as png files."""
    xlspath = Path('/run/media/tzu-yu/linuxData/Research/PriA_project/analysis_result/20191106//20191106parameterFile.xlsx')
    sheet = 'L1_03'
    dfs = utils.read_excel(xlspath, sheet_name=sheet)
    datapath = imscrollIO.def_data_path()
    im_format = 'png'
    nFiles = dfs.shape[0]
    for iFile in range(0, nFiles):
        filestr = dfs.filename[iFile]

        try:
            all_data, AOI_categories = binding_kinetics.load_all_data(datapath
                                                                      / (filestr + '_all.json'))
        except FileNotFoundError:
            print('{} file not found'.format(filestr))
            continue
        data = all_data['data']
        print(filestr + ' loaded')
        save_dir = datapath / filestr
        if not save_dir.is_dir():
            save_dir.mkdir()
        AOI_dict = dict(AOI_categories)
        for key in AOI_categories:
            if key == 'analyzable':
                del AOI_dict[key]
                for key2, value2 in AOI_categories[key].items():
                    AOI_dict[key2] = value2

        for key, AOI_list in AOI_dict.items():
            for iAOI in AOI_list:
                molecule_data = data.sel(AOI=iAOI)
                plot_one_trace_and_save(molecule_data, key,
                                        save_dir=save_dir, save_format=im_format)


def plot_scatter_and_linear_fit(x, y, fit_result: dict,
                                save_fig_path: Path = None,
                                x_label: str = '',
                                y_label: str = '',
                                left_bottom_as_origin=False,
                                y_top=None,
                                x_right=None):
    fig, ax = plt.subplots()
    ax.scatter(x, y)
    line_x = np.linspace(min(x), max(x), 10)
    line_y = line_x * fit_result['slope'] + fit_result['intercept']
    ax.plot(line_x, line_y)
    if left_bottom_as_origin:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    if y_top is not None:
        ax.set_ylim(top=y_top)
    if x_right is not None:
        ax.set_xlim(right=x_right)
    if not x_label:
        x_label = input('Enter x axis label:\n')
    if not y_label:
        y_label = input('Enter y axis label:\n')
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.text(0.1, 0.9, r'$y = {:.7f}x + {:.5f}$'.format(fit_result['slope'],
                                                       fit_result['intercept']),
            transform=ax.transAxes, fontsize=12)
    ax.text(0.1, 0.8, r'$R^2 = {:.5f}$'.format(fit_result['r_squared']),
            transform=ax.transAxes, fontsize=12)
    plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(save_fig_path, format='svg', dpi=300, bbox_inches='tight')


def plot_error_and_linear_fit(x, y, y_err, fit_result: dict,
                                save_fig_path: Path = None,
                                x_label: str = '',
                                y_label: str = '',
                                left_bottom_as_origin=False,
                                y_top=None,
                                x_right=None,
                              x_raw=None,
                              y_raw=None):
    sns.set_palette(palette='muted')
    np.random.seed(0)
    fig, ax = plt.subplots(figsize=(4, 3))

    sns.despine(fig, ax)
    # ax.set_xticks(sorted(x))
    ax.errorbar(x, y, yerr=y_err, marker='o', ms=5, linestyle='', capsize=2)
    line_x = np.array([min(x), max(x)])
    line_y = line_x * fit_result['slope'] + fit_result['intercept']
    ax.plot(line_x, line_y, zorder=0)
    x_jitter = 0.01 * (x_raw.max()-x_raw.min()) * np.random.standard_normal(x_raw.shape)
    ax.scatter(x=x_raw + x_jitter, y=y_raw, marker='o', color='w', edgecolors='gray', linewidth=1, s=14, zorder=3)
    if left_bottom_as_origin:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    if y_top is not None:
        ax.set_ylim(top=y_top)
    if x_right is not None:
        ax.set_xlim(right=x_right)
    if not x_label:
        x_label = input('Enter x axis label:\n')
    if not y_label:
        y_label = input('Enter y axis label:\n')
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.text(0.05, 0.9, r'$y = {:.7f}x + {:.5f}$'.format(fit_result['slope'],
                                                       fit_result['intercept']),
            transform=ax.transAxes, fontsize=8)
    ax.text(0.05, 0.8, r'$R^2 = {:.5f}$'.format(fit_result['r_squared']),
            transform=ax.transAxes, fontsize=8)
    # plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(save_fig_path, format='svg', dpi=300, bbox_inches='tight')



def plot_error(x, y, y_err,
                                save_fig_path: Path = None,
                                x_label: str = '',
                                y_label: str = '',
                                left_bottom_as_origin=False,
                                y_top=None,
                                x_right=None,
                              x_raw=None,
                              y_raw=None):
    sns.set_palette(palette='muted')
    np.random.seed(0)
    fig, ax = plt.subplots(figsize=(4, 3))

    sns.despine(fig, ax)
    # ax.set_xticks(sorted(x))
    ax.errorbar(x, y, yerr=y_err, marker='o', ms=5, linestyle='', capsize=2)
    x_jitter = 0.01 * (x_raw.max()-x_raw.min()) * np.random.standard_normal(x_raw.shape)
    ax.scatter(x=x_raw + x_jitter, y=y_raw, marker='o', color='w', edgecolors='gray', linewidth=1, s=14, zorder=3)
    if left_bottom_as_origin:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
    if y_top is not None:
        ax.set_ylim(top=y_top)
    if x_right is not None:
        ax.set_xlim(right=x_right)
    if not x_label:
        x_label = input('Enter x axis label:\n')
    if not y_label:
        y_label = input('Enter y axis label:\n')
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    # plt.rcParams['svg.fonttype'] = 'none'
    fig.savefig(save_fig_path, format='svg', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
