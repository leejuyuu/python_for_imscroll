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
    print(image.shape)


    ## Always start by initializing Qt (only once per application)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    imv = pg.ImageView()
    # imv = QtWidgets.QWidget()
    imv.show()
    imv.setImage(image, axes={'t': 2, 'x': 1, 'y': 0})


    ## Start the Qt event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
