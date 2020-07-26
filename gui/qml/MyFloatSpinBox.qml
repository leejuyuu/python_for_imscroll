import QtQuick 2.15
import MySpinBox

MySpinBox {
    id: control

    value: realValue * 10

    validator: DoubleValidator {
        bottom: Math.min(control.from, control.to)
        top:  Math.max(control.from, control.to)
    }
    property int decimals: 1
    property real realValue: model.display
    textFromValue: function(value, locale) {
        return Number(value /10 ).toLocaleString(locale, 'f', control.decimals)
    }

    valueFromText: function(text, locale) {
        return Number.fromLocaleString(locale, text) * 10
    }
}
