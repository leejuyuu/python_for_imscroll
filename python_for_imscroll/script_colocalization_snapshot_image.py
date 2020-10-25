
from pathlib import Path
import skimage.exposure
import numpy as np
import matplotlib.pyplot as plt
import python_for_imscroll.image_processing as imp

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
    datapath = Path('/run/media/tzu-yu/linuxData/Research/PriA_project/analysis_result/20200228/20200228imscroll/')
    image_path = Path('/run/media/tzu-yu/linuxData/Research/PriA_project/0228/L2_GstPriA_125pM/hwligroup00774/')
    filestr = 'L2'
    image_sequence = imp.ImageSequence(image_path)
    aois = imp.Aois.from_imscroll_aoiinfo2(datapath / (filestr + '_aoi.dat'))
    fig, ax = plt.subplots(figsize=(4, 2))
    image = image_sequence.get_averaged_image(0, 10)
    scale = quickMinMax(image)
    ax.imshow(image, cmap='gray', vmin=scale[0], vmax=scale[1], interpolation='nearest')
    ax.set_axis_off()
    ax.scatter(aois.get_all_x(),
               aois.get_all_y(),
               marker='s',
               color='none',
               edgecolors='yellow',
               linewidth=1,
               s=100,
               )
    origin = np.array([270, 125]) - 0.5 # Offset by 0.5 to the edge of pixel
    size = 50
    ax.set_xlim((origin[0], origin[0] + size))
    ax.set_ylim((origin[1], origin[1] + size))
    # ax.xaxis.set_visible(False)
    # ax.yaxis.set_visible(False)
    fig.savefig(Path('~/image_temp.svg').expanduser(), dpi=300, format='svg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == '__main__':
    main()
