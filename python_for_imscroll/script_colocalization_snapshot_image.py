
from pathlib import Path
import skimage.exposure
import numpy as np
import matplotlib.pyplot as plt
import python_for_imscroll.image_processing as imp
from python_for_imscroll import mapping

def quickMinMax( data):
    """
    Estimate the min/max values of *data* by subsampling.
    Returns [(min, max), ...] with one item per channel

    Copied from the pyqtgraph ImageView class
    """
    while data.size > 1e6:
        ax = np.argmax(data.shape)
        sl = [slice(None)] * data.ndim
        sl[ax] = slice(None, None, 2)
        data = data[tuple(sl)]

    if data.size == 0:
        return [(0, 0)]
    return (float(np.nanmin(data)), float(np.nanmax(data)))

def main():
    datapath = Path('~/Analysis_Results/20200928/20200928imscroll/').expanduser()
    image_path = Path('~/Expt_data/20200928/L4_GstPriA_500pM/hwligroup01310/').expanduser()
    image_path_green = Path('~/Expt_data/20200928/L4_GstPriA_500pM/hwligroup01311/').expanduser()

    filestr = 'L4_20'
    image_sequence = imp.ImageSequence(image_path)
    aois = imp.Aois.from_imscroll_aoiinfo2(datapath / (filestr + '_aoi.dat'))
    aois.channel = 'blue'
    fig, ax = plt.subplots(figsize=(2, 2))
    image = image_sequence.get_averaged_image(20, 1)
    scale = quickMinMax(image)
    ax.imshow(image, cmap='gray', vmin=scale[0], vmax=scale[1], interpolation='nearest', origin='upper')
    ax.set_axis_off()
    ax.scatter(aois.get_all_x(),
               aois.get_all_y(),
               marker='s',
               color='none',
               edgecolors='yellow',
               linewidth=1,
               s=100,
               )
    origin = np.array([320, 50]) - 0.5 # Offset by 0.5 to the edge of pixel
    size = 80
    ax.set_xlim((origin[0], origin[0] + size))
    ax.set_ylim((origin[1], origin[1] + size))
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    fig.savefig(Path('~/image_temp_blue.svg').expanduser(), dpi=1200, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    mapping_file_path = Path('/run/media/tzu-yu/main/git_repos/Imscroll-and-Utilities/data/mapping/20200803_bg_6.dat')
    mapper = mapping.Mapper.from_imscroll(mapping_file_path)
    mapped_aois = mapper.map(aois, to_channel='green')


    image_sequence = imp.ImageSequence(image_path_green)
    aois = mapped_aois
    aois.to_npz(Path('~/123').expanduser())
    fig, ax = plt.subplots(figsize=(2, 2))
    image = image_sequence.get_averaged_image(20, 1)
    scale = quickMinMax(image)
    ax.imshow(image, cmap='gray', vmin=scale[0], vmax=scale[1], interpolation='nearest', origin='upper')
    ax.set_axis_off()
    ax.scatter(aois.get_all_x(),
               aois.get_all_y(),
               marker='s',
               color='none',
               edgecolors='yellow',
               linewidth=0.4,
               s=100,
               linestyle=':')
    ref_aoi = imp.pick_spots(image_sequence.get_averaged_image(20, 2),
                             230)
    colocalized_aois = aois.remove_aois_far_from_ref(ref_aoi, 1.5)
    ax.scatter(colocalized_aois.get_all_x(),
               colocalized_aois.get_all_y(),
               marker='s',
               color='none',
               edgecolors='yellow',
               linewidth=1,
               s=100,
               )
    origin += mapper.map_matrix[('blue', 'green')][:, 2]
    # size = 60
    ax.set_xlim((origin[0], origin[0] + size))
    ax.set_ylim((origin[1], origin[1] + size))
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    fig.savefig(Path('~/image_temp_green.svg').expanduser(), dpi=1200, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == '__main__':
    main()
