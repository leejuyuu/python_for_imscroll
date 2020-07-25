from collections import namedtuple
from pathlib import Path
import sys
import typing
import numpy as np
import pyqtgraph as pg
from PySide2 import QtWidgets, QtQuickWidgets, QtCore, QtQml
from PySide2.QtCore import Qt
import python_for_imscroll.image_processing as imp

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

SPOT_DIA_STR = 'Spot diameter'
NOISE_DIA_STR = 'Noise diameter'
SPOT_BRIGHTNESS_STR = 'Spot brightness'
ValueRange = namedtuple('ValueRange', ('min', 'max', 'step'))
SPOT_PARAMS_RANGE = {SPOT_DIA_STR: ValueRange(0, 100, 0.5),
                     NOISE_DIA_STR: ValueRange(0, 10, 0.2),
                     SPOT_BRIGHTNESS_STR: ValueRange(0, 1000, 1)}

PROPERTY_NAME_ROLE = Qt.UserRole + 1

class MyImageView(pg.ImageView):

    def __init__(self):
        super().__init__()
        self.view_box = self.getView()
        self.model = Model()
        self.aois_view = AoisView(self.view_box, self.model)
        self.view_box.setAspectLocked(lock=True)
        self.aois_view.pick_aois.connect(self.model.pick_spots, QtCore.Qt.UniqueConnection)
        self.model.aois_changed.connect(self.aois_view.update, QtCore.Qt.UniqueConnection)
        self.sigTimeChanged.connect(self.model.change_current_frame, QtCore.Qt.UniqueConnection)

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
        self._current_frame = 0
        self.pick_spots_param = PickSpotsParam()


    def get_coords(self):
        if self.aois is None:
            # return (None, None)
            return (np.empty(1), np.empty(1))
        return (self.aois.get_all_x(), self.aois.get_all_y())

    def set_aois(self, aois):
        self.aois = aois

    def pick_spots(self):
        params = self.pick_spots_param.params
        self.aois = imp.pick_spots(self.image_sequence.get_averaged_image(start=self._current_frame, size=1),
                                   threshold=params[SPOT_BRIGHTNESS_STR],
                                   noise_dia=params[NOISE_DIA_STR],
                                   spot_dia=params[SPOT_DIA_STR])
        self.aois_changed.emit()

    @QtCore.Slot(int)
    def change_current_frame(self, new_frame_index: int):
        self._current_frame = new_frame_index

    @property
    def current_frame(self):
        return self._current_frame

    @current_frame.setter
    def current_frame(self, new_value: int):
        new_value = int(new_value)
        if new_value > len(self.image_sequence):
            new_value = len(self.image_sequence)
        elif new_value < 0:
            new_value = 0
        self._current_frame = new_value

    def _read_pick_spots_param(self):
        return self.pick_spots_param

    @QtCore.Signal
    def dummy_notify(self):
        pass

    pickSpotsParam = QtCore.Property(QtCore.QObject,
                                     fget=_read_pick_spots_param,
                                     notify=dummy_notify)

class PickSpotsParam(QtCore.QAbstractListModel):
    def __init__(self):
        super().__init__()
        NV = namedtuple('NV', ['name', 'value'])
        self.property_names = [SPOT_DIA_STR, NOISE_DIA_STR, SPOT_BRIGHTNESS_STR]
        self.params = {SPOT_DIA_STR: 5,
                       NOISE_DIA_STR: 1,
                       SPOT_BRIGHTNESS_STR: 50}

    def roleNames(self):
        """See base class."""
        role_names = super().roleNames()
        role_names[PROPERTY_NAME_ROLE] = b'propertyName'
        role_names[Qt.UserRole + 10] = b'min'
        role_names[Qt.UserRole + 11] = b'max'
        role_names[Qt.UserRole + 12] = b'step'
        return role_names

    def rowCount(self, parent: QtCore.QModelIndex = None) -> int:
        """See base class."""
        return len(self.params)

    def data(self, index: QtCore.QModelIndex, role: int = Qt.DisplayRole):
        """See base class."""
        name = self.property_names[index.row()]
        if role == Qt.DisplayRole:
            return self.params[name]
        if role == Qt.EditRole:
            return self.params[name]
        if role == PROPERTY_NAME_ROLE:
            return name
        if role == Qt.UserRole + 10:
            return SPOT_PARAMS_RANGE[name].min
        if role == Qt.UserRole + 11:
            return SPOT_PARAMS_RANGE[name].max
        if role == Qt.UserRole + 12:
            return SPOT_PARAMS_RANGE[name].step

    def setData(self, index: QtCore.QModelIndex, value: typing.Any,
                role: int = None) -> bool:
        """See base class."""
        if index.isValid() and role == Qt.EditRole:
            name = self.property_names[index.row()]
            self.params[name] = value
            return True
        return False

    def flags(self, index: QtCore.QModelIndex) -> Qt.ItemFlags:
        """See base class."""
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEditable | Qt.ItemIsEnabled


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
    def update(self):
        coords = self.model.get_coords()
        # The coordinate of the view starts from the edge, so offsets 0.5
        self.marker.setData(coords[0] + 0.5, coords[1] + 0.5, symbol='s', pen=(0, 0, 255), brush=None, size=5, pxMode=False)

class Window(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        layout = QtWidgets.QGridLayout()
        self.image_view = MyImageView()
        self.resize(640, 480)

        self.qml = QtQuickWidgets.QQuickWidget()
        # self.qml.setResizeMode(QtQuickWidgets.QQuickWidget.SizeRootObjectToView)
        qml_path = Path(__file__).parent / 'image_view.qml'
        root_context = self.qml.rootContext()
        root_context.setContextProperty('imageView', self.image_view)
        root_context.setContextProperty('dataModel', self.image_view.model)
        # root_context.setContextProperty('hhh', self.image_view.model.pickSpotsParam)
        # Need to set context property before set source
        self.qml.setSource(QtCore.QUrl(str(qml_path)))

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
    app.exec_()
    model = window.image_view.model # This line here is to make sure the context property lives before qml quits
    # To avoid a warning thrown by the qml
    # cannot think of a better way right now
    sys.exit()


if __name__ == '__main__':
    main()
