from pathlib import Path
import scipy.io as sio
import numpy as np
from skimage import exposure
import skimage.io
from python_for_imscroll import binding_kinetics
import python_for_imscroll.image_processing as imp
import python_for_imscroll.script_colocalization_snapshot_image as smodule


def load_image_one_frame(frame, header_file_path):
    header_file = sio.loadmat(header_file_path)
    offset = header_file['vid']['offset'][0, 0].squeeze()
    file_number = header_file['vid']['filenumber'][0, 0].squeeze()
    width = header_file['vid']['width'][0, 0].item()
    height = header_file['vid']['height'][0, 0].item()
    print(width, height)
    n_pixels = width * height
    image_file_path = header_file_path.parent / '{}.glimpse'.format(file_number[frame - 1])
    dt = np.dtype('>i2')
    print(dt.name)
    
    arr = np.fromfile(image_file_path, dtype='>i2', count=n_pixels, offset=offset[frame - 1])
    print(arr.shape)
    arr2 = np.reshape(arr, (height, width))
    arr2 = np.transpose(arr2)
    print(arr2.shape)
    arr2 += 2**15
    print(arr2[0])
    return arr2

def read_coordinate_sequences(int_corrected_path, channel):
    file = sio.loadmat(int_corrected_path)
    aoifits_array = file['aoifits']['data' + channel][0, 0]
    coords = aoifits_array[:, [3, 4]]
    n_frames = int(max(aoifits_array[:, 1]))
    n_aoi = int(max(aoifits_array[:, 0]))
    coords = np.reshape(coords, (n_frames, n_aoi, 2))
    # print(coords[0])
    coords = np.swapaxes(coords, 1, 2)
    # breakpoint()
    return coords



def main():
    aoi = 7
    # datapath = Path('/run/media/tzu-yu/linuxData/Research/PriA_project/analysis_result/20200228/20200228imscroll/')
    datapath = Path('/run/media/tzu-yu/data/PriA_project/Analysis_Results/20200317/20200317imscroll/')
    # image_path = Path('/run/media/tzu-yu/linuxData/Research/PriA_project/0228/L2_GstPriA_125pM/hwligroup00775/')
    image_path = Path('/run/media/tzu-yu/data/PriA_project/Expt_data/20200317/L5_GstPriA_125pM/L5_02_photobleaching_03/hwligroup00821/')
    image_sequence = imp.ImageSequence(image_path)
    filestr = 'L5_02_03'
    framestart = 0
    int_corrected_path = datapath / (filestr + '_intcorrected.dat')
    try:
        all_data, AOI_categories = binding_kinetics.load_all_data(datapath
                                                                    / (filestr + '_all.json'))
    except FileNotFoundError:
        print('{} file not found'.format(filestr))
    channel = 'green'
    channel_data = all_data['data'].sel(channel=channel)
    green_coord = read_coordinate_sequences(int_corrected_path, channel)
    green_coord_aoi = green_coord[:, :, aoi - 1]
    interval = all_data['intervals']
    aoi_interval = interval.sel(AOI=aoi)
    aoi_interval = aoi_interval.dropna(dim='interval_number')

    state_sequence = channel_data['viterbi_path'].sel(state='label', AOI=aoi)
    state_start_index = binding_kinetics.find_state_end_point(state_sequence)
    #print(len(state_start_index))
    if state_start_index.any():
        for event_end in state_start_index:
            spacer = 1
            out = np.zeros((11, 11*6+5*spacer), dtype='uint16') - 1
            print(out[0])
            spacer_list = []
            a = np.arange(spacer)
            for i in range(5):
                a += 11
                spacer_list.extend(a.tolist())
                a += spacer
            
            
            out[:, spacer_list] = 150 * 2**8
            for i, frame in enumerate(range(event_end-2, event_end + 4)):
                coord = np.round(green_coord_aoi[frame, :]) - 1
                # img = load_image_one_frame(frame + 1, header_file_path)
                img = image_sequence.get_one_frame(frame+framestart)
                scale = smodule.quickMinMax(img)
                # print(scale)
                scale = (500, 3000)
                dia = 11
                y = int(coord[0])
                x = int(coord[1])
                rad = int((dia-1)/2)
                # print(x, y)
                # print(img.shape)
                sub_img = img[y - rad:y + rad + 1,
                              x - rad:x + rad + 1]

                im = exposure.rescale_intensity(sub_img,
                                           in_range=scale,
                                           out_range=(2**13, 2**16-1))
                #print(im.dtype)
                # im = exposure.rescale_intensity(im,
                #                            out_range=(30, 255))
                # # print(sub_img.shape)
                arr = np.zeros(sub_img.shape, dtype='uint8')
                # print(arr.shape)
                arr[:, :] = im
                out[:, i * (11 + spacer):i * (11 + spacer) + 11] = im
            save_path = datapath / '{}_{}_{:.0f}.png'.format(filestr,
                                                                aoi,
                                                                channel_data.time[event_end].item()+24.226)
            out = skimage.util.invert(out)
           # print(out)
            skimage.io.imsave(save_path, out)



if __name__ == '__main__':
    main()
