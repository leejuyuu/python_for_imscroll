import QtQuick 2.15
import QtQuick.Controls 2.15

Item{
    id: root
    width: 300; height: 480

    Button {
        id: pickButton
        width: 60
        text: 'Pick'
        onClicked: imageView.onPickButtonPressed()

    }
    Button {
        id: fitButton
        width: 60
        anchors {
            left: pickButton.right
        }
        text: 'Fit'
        onClicked: imageView.onFitButtonPressed()
    }
    Button {
        id: addButton
        width: 60
        anchors {
            left: fitButton.right
        }
        text: 'Add'
        onClicked: imageView.onAddButtonPressed()
    }
    Button {
        id: removeButton
        width: 60
        anchors {
            left: addButton.right
        }
        text: 'Remove'
        onClicked: imageView.onRemoveButtonPressed()
    }

    ListView {
        anchors {
            left: parent.left
            right: parent.right
            top: pickButton.bottom
            bottom: parent.bottom
        }
        model: dataModel.pickSpotsParam
        interactive: false
        spacing: 20
        delegate: Item {
            id: entryRoot
            implicitHeight: 30
            anchors {
                left: parent.left; right: parent.right
            }
            Item {
                id: entryTextRegion
                anchors {
                    fill: parent
                    leftMargin: 5
                    rightMargin: parent.width - 120
                }
                Text {
                    anchors.fill: parent
                    text: model.propertyName
                    verticalAlignment: Text.AlignVCenter
                }
            }
            MySpinBox {
                from: model.min
                to: model.max
                stepSize: model.step
                anchors {
                    left: parent.left
                    leftMargin: 120
                }
                value: model.display
                onValueModified: {model.edit = value}
            }
        }
    }

}