import QtQuick 2.15
import QtQuick.Controls 2.15


SpinBox {
    id: control
    implicitWidth: 80
    down.indicator.implicitHeight: 20
    down.indicator.implicitWidth: 20
    up.indicator.implicitHeight: 20
    up.indicator.implicitWidth: 20
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

    editable: true
    validator: IntValidator {
            locale: control.locale.name
            bottom: Math.min(control.from, control.to)
            top: Math.max(control.from, control.to)
    }

}
