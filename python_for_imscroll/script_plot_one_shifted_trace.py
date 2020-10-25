from pathlib import Path
import matplotlib.pyplot as plt
from python_for_imscroll import binding_kinetics, visualization

def main():
    plt.style.use(str(Path('./trace_style.mplstyle').resolve()))
    datapath = Path('/run/media/tzu-yu/linuxData/Research/PriA_project/analysis_result/20200922/20200922imscroll/')
    filestr = 'L1'
    aois = [49, 71]
    time_offset = 24.226
    save_dir = datapath / filestr
    try:
        all_data, _ = binding_kinetics.load_all_data(datapath
                                                     / (filestr + '_all.json'))
    except FileNotFoundError:
        print('{} file not found'.format(filestr))

    channel_data = all_data['data'].sel(channel='green')
    fig, ax_list = plt.subplots(nrows=len(aois))
    for molecule, ax in zip(aois, ax_list):
        intensity = channel_data['intensity'].sel(AOI=molecule)
        vit = channel_data['viterbi_path'].sel(AOI=molecule, state='position')
        ax.plot(intensity.time + time_offset, intensity, color='#017517')
        ax.plot(vit.time + time_offset, vit, color='black')
        if ax.get_ylim()[0] > 0:
            ax.set_ylim(bottom=0)
        ax.set_ylim(ax.get_ylim())
        if time_offset:
            ax.fill([0, time_offset, time_offset, 0],
                    [ax.get_ylim()[0], ax.get_ylim()[0], ax.get_ylim()[1], ax.get_ylim()[1]],
                    '#e4e4e4')
        ax.set_xlim((0, intensity.time.max() + time_offset))
        # ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 2), useMathText=True)
    ax_list[0].xaxis.set_visible(False)
    # ax = fig.add_subplot(1, 1, 1)

    fig.text(0.04, 0.4, 'Intensity', ha='center', rotation='vertical')
    ax.set_xlabel('Time (s)')
    if time_offset:
        plt.savefig(save_dir / ('molecule{},{}_shifted.{}'.format(*aois, '.svg')),
                    format='svg')
    else:
        plt.savefig(save_dir / ('molecule{},{}.{}'.format(*aois, '.svg')),
                     format='svg')
    plt.close()

if __name__ == '__main__':
    main()
