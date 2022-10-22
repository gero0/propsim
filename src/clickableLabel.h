#pragma once

#include <QtWidgets>

class ClickableLabel : public QLabel {
    Q_OBJECT

signals:
    void displayGridClicked(QMouseEvent* myEvent); // Signal to emit

protected:
    bool event(QEvent* myEvent); // This method will give all kind of events on Label Widget
public:
    ClickableLabel(QWidget* parent);
};