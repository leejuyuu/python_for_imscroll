from pathlib import Path
import pandas as pd
import imscrollIO
import binding_kinetics as bk
import binding_kinetics
xlspath = Path('D:/TYL/PriA_project/Analysis_Results/20190917/20190917parameterFile.xlsx')
# xlspath = Path('/mnt/linuxData/Research/PriA_project/analysis_result/20190911/20190911parameterFile.xlsx')

datapath = imscrollIO.def_data_path()
# datapath = Path('/mnt/linuxData/Research/PriA_project/analysis_result/20190911/20190911imscroll')
sheet_list = ['L1', 'L2', 'L3', 'L4']
for i_sheet in sheet_list:
    dfs = pd.read_excel(xlspath, sheet_name=i_sheet)
    nFiles = dfs.shape[0]
    for iFile in range(0, nFiles):
        filestr = dfs.filename[iFile]
        AOI_categories = {}

        try:
            data = imscrollIO.load_data_from_json(datapath / (filestr + '_data.json'))
        except FileNotFoundError:
            continue

        state_info = bk.collect_all_channel_state_info(data)
        DNA_channel_data = bk.get_channel_data(data, data.target_channel)
        bad_tethers = bk.list_multiple_DNA(state_info.sel(channel=data.target_channel))
        bad_tethers_list = list(bad_tethers)
        bad_tethers_list.sort()
        AOI_categories['multiple_tethers'] = bad_tethers_list
        selected_data \
            = bk.split_data_set_by_specifying_aoi_subset(data, bad_tethers)[1]
        selected_state_info = \
            bk.split_data_set_by_specifying_aoi_subset(state_info, bad_tethers)[1]

        protein_channel_data = bk.get_channel_data(selected_data, data.binder_channel[0])
        bad_aoi_set, intervals = bk.match_vit_path_to_intervals(
            protein_channel_data)
        bad_aoi_list = list(bad_aoi_set)
        bad_aoi_list.sort()

        AOI_categories['false_binding'] = bad_aoi_list
        selected_data \
            = bk.split_data_set_by_specifying_aoi_subset(selected_data, bad_aoi_set)[1]
        good_intervals = \
            bk.split_data_set_by_specifying_aoi_subset(intervals, bad_aoi_set)[1]

        AOI_categories['analyzable'] = \
            bk.group_analyzable_aois_into_state_number(selected_data, good_intervals)
        all_data = {'data': data,
                    'intervals': intervals,
                    'state_info': state_info}
        bk.save_all_data(all_data, AOI_categories, datapath / (filestr + '_all.json'))

        print(filestr + ' finished')
        123
    
    #
    # nAOIs = len(data.AOI)
    #
    #
    # if not os.path.isdir(datapath / filestr):
    #     os.mkdir(datapath / filestr)
    # for iAOI in data.AOI:
    #
    #     plt.figure(figsize=(10, 5))
    #
    #     plt.suptitle('molecule {}'.format(iAOI.values), fontsize=14)
    #     plt.xlim((0, data.time.max()))
    #     plt.xlabel('time', fontsize=16)
    #
    #     plt.ylabel('Intensity', fontsize=16)
    #     for i, i_channel in enumerate(data.channel.values, 1):
    #
    #         plt.subplot(3, 1, i)
    #         y = data['intensity'].sel(AOI=iAOI, channel=i_channel)
    #         bool_nan = xr.ufuncs.logical_not(xr.ufuncs.isnan(y))
    #         plt.plot(data.time.where(bool_nan, drop=True),
    #                  y.where(bool_nan, drop=True),
    #                  color=i_channel
    #                  )
    #         plt.xlim((0, data.time.max()))
    #
    #
    #
    #     plt.xlabel('time (s)', fontsize=16)
    #
    #     plt.ylabel('Intensity', fontsize=16)
    #     # plt.show()
    #
    #
    #
    #
    #     plt.savefig(datapath / filestr / ('molecule{}.png'.format(iAOI.values)), Transparent=True,
    #                 dpi=300, bbox_inches='tight')
    #
    #     plt.close()

123