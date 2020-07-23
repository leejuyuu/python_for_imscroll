from pathlib import Path
import sys
import numpy as np
import pyqtgraph as pg
from PySide2 import QtWidgets, QtQuickWidgets, QtCore
import python_for_imscroll.image_processing as imp

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

class MyImageView(pg.ImageView):

    def __init__(self):
        super().__init__()
        self.view_box = self.getView()
        self.model = Model()
        self.aois_view = AoisView(self.view_box, self.model)
        self.view_box.setAspectLocked(lock=True)
        self.aois_view.pick_aois.connect(self.model.pick_spots, QtCore.Qt.UniqueConnection)
        self.model.aois_changed.connect(self.aois_view.update, QtCore.Qt.UniqueConnection)

    def setSequence(self, image_sequence: imp.ImageSequence):
        self.imageSequence = image_sequence
        image = image_sequence.get_whole_stack()
        self.view_box.setLimits(xMin=0, xMax=image_sequence.width, yMin=0, yMax=image_sequence.height)
        self.setImage(image, axes={'t': 2, 'x': 1, 'y': 0})
        self.view_box.setRange(yRange=(0, image_sequence.height))


    @QtCore.Slot()
    def onPickButtonPressed(self):
        self.aois_view.pick_aois.emit()

class Model(QtCore.QObject):

    aois_changed = QtCore.Signal()
    def __init__(self):
        super().__init__()
        self.aois: imp.Aois = None
        self.image_sequence: imp.ImageSequence = None

    def get_coords(self):
        if self.aois is None:
            # return (None, None)
            return (np.empty(1), np.empty(1))
        return (self.aois.get_all_x(), self.aois.get_all_y())

    def set_aois(self, aois):
        self.aois = aois

    def pick_spots(self):
        self.aois = imp.pick_spots(self.image_sequence.get_averaged_image(size=10),
                                   threshold=50,
                                   noise_dia=1,
                                   spot_dia=5)
        self.aois_changed.emit()

class AoisView(QtCore.QObject):
    pick_aois = QtCore.Signal()
    def __init__(self, view_box, model):
        super().__init__()
        self.marker = pg.ScatterPlotItem()
        self.marker.setBrush(255, 0, 0, 255)
        self.model = model
        self.update()
        view_box.addItem(self.marker)

    @QtCore.Slot()
    def sss(self):
        print(123)

    @QtCore.Slot()
    def update(self):
        coords = self.model.get_coords()
        # The coordinate of the view starts from the edge, so offsets 0.5
        self.marker.setData(coords[0] - 0.5, coords[1] - 0.5, symbol='s', pen=(0, 0, 255), brush=None, size=5, pxMode=False)

class Window(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QGridLayout()
        self.image_view = MyImageView()
        self.resize(640, 480)

        self.qml = QtQuickWidgets.QQuickWidget()
        # self.qml.setResizeMode(QtQuickWidgets.QQuickWidget.SizeRootObjectToView)
        qml_path = Path(__file__).parent / 'image_view.qml'
        self.qml.setSource(QtCore.QUrl(str(qml_path)))
        root_context = self.qml.rootContext()
        root_context.setContextProperty('imageView', self.image_view)

        layout.addWidget(self.image_view)
        layout.addWidget(self.qml, 0, 1)
        layout.setColumnStretch(0, 4)
        layout.setColumnStretch(1, 1)
        self.setLayout(layout)

def main():
    path = Path('/run/media/tzu-yu/linuxData/Git_repos/python_for_imscroll/python_for_imscroll/test/test_data/fake_im/')
    path = Path('/run/media/tzu-yu/linuxData/Research/PriA_project/20200206_mapping/map/hwligroup00688/')
    image_sequence = imp.ImageSequence(path)
    image = image_sequence.get_whole_stack()


    ## Always start by initializing Qt (only once per application)
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])

    window = Window()
    # imv = MyImageView()
    window.image_view.setSequence(image_sequence)
    window.image_view.model.image_sequence = image_sequence
    window.show()



    ## Start the Qt event loop
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
