from pathlib import Path
from python_for_imscroll import binding_kinetics, visualization

def main():
    datapath = Path('/run/media/tzu-yu/linuxData/Research/PriA_project/analysis_result/20200228/20200228imscroll/')
    filestr = 'L2'
    aoi = 22
    time_offset = 34
    savedir = datapath / filestr
    try:
        all_data, AOI_categories = binding_kinetics.load_all_data(datapath
                                                                    / (filestr + '_all.json'))
    except FileNotFoundError:
        print('{} file not found'.format(filestr))

    molecule_data = all_data['data'].sel(AOI=aoi)
    visualization.plot_one_trace_and_save(molecule_data, save_dir=savedir,
                                          time_offset=time_offset)

if __name__ == '__main__':
    main()
