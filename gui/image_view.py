from pathlib import Path
import sys
import pyqtgraph as pg
from PySide2 import QtWidgets
import python_for_imscroll.image_processing as imp

def main():
    path = Path('/run/media/tzu-yu/linuxData/Git_repos/python_for_imscroll/python_for_imscroll/test/test_data/fake_im/')
    path = Path('/run/media/tzu-yu/linuxData/Research/PriA_project/20200206_mapping/map/hwligroup00688/')
    image_sequence = imp.ImageSequence(path)
    image = image_sequence.get_whole_stack()


    ## Always start by initializing Qt (only once per application)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    vb = pg.ViewBox()
    vb.setLimits(xMin=0, xMax=image_sequence.width, yMin=0, yMax=image_sequence.height)
    imv = pg.ImageView(view=vb)
    marker = pg.ScatterPlotItem()
    marker.setBrush(255, 0, 0, 255)
    spot_dia = 5
    noise_dia = 1
    spot_b = 80
    filtered_image = imp.band_pass(image[:, :, 0], noise_dia, spot_dia)
    peaks = imp.find_peaks(filtered_image, spot_b, spot_dia)
    # The coordinate of the view starts from the edge, so offsets 0.5
    peaks = imp.localize_centroid(filtered_image, peaks, spot_dia + 2) - 0.5
    marker.setData(peaks[:, 0], peaks[:, 1], symbol='s', pen=(0, 0, 255), brush=None, size=5, pxMode=False)
    vb.addItem(marker)
    # imv = QtWidgets.QWidget()
    imv.show()
    imv.setImage(image, axes={'t': 2, 'x': 1, 'y': 0})



    ## Start the Qt event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
