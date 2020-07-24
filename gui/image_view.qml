import QtQuick 2.15
import QtQuick.Controls 2.15

Item{
    id: root
    width: 300; height: 480

    Button {
        id: button1
        text: 'Pick'
        onClicked: imageView.onPickButtonPressed()

    }

    ListView {
        anchors {
            left: parent.left
            right: parent.right
            top: button1.bottom
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
            SpinBox {
                id: control
                implicitWidth: 80
                down.indicator.implicitHeight: 20
                down.indicator.implicitWidth: 20
                up.indicator.implicitHeight: 20
                up.indicator.implicitWidth: 20
                from: model.min
                to: model.max
                anchors {
                    left: parent.left
                    leftMargin: 120
                }
                contentItem: TextInput {
                    // This item was copied from the source code, I only needed
                    // to modify its selectByMouse property
                    z: 2
                    text: control.displayText

                    font: control.font
                    color: control.palette.text
                    selectionColor: control.palette.highlight
                    selectedTextColor: control.palette.highlightedText
                    horizontalAlignment: Qt.AlignHCenter
                    verticalAlignment: Qt.AlignVCenter

                    readOnly: !control.editable
                    validator: control.validator
                    inputMethodHints: control.inputMethodHints
                    selectByMouse: true

                    Rectangle {
                        x: -6 - (control.down.indicator ? 1 : 0)
                        y: -6
                        width: control.width - (control.up.indicator ? control.up.indicator.width - 1 : 0) - (control.down.indicator ? control.down.indicator.width - 1 : 0)
                        height: control.height
                        visible: control.activeFocus
                        color: "transparent"
                        border.color: control.palette.highlight
                        border.width: 2
                    }
                }

                value: model.display
                editable: true
                onValueModified: {model.edit = value}

            }
        }
    }

}
