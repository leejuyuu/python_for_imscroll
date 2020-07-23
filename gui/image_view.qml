import QtQuick 2.15
import QtQuick.Controls 2.15

Item{
    id: root
    width: 200; height: 480

    Button {
        id: button1
        text: 'Pick'
        onClicked: imageView.onPickButtonPressed()

    }
}
