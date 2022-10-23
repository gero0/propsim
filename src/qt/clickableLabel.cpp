#include "clickableLabel.h"
#include <qevent.h>

ClickableLabel::ClickableLabel() : QLabel(){

}

ClickableLabel::ClickableLabel(QWidget* parent)
    : QLabel(parent)
{
}

bool ClickableLabel::event(QEvent* myEvent)
{
    switch (myEvent->type()) {
    case (QEvent ::MouseButtonRelease): // Identify Mouse press Event
    {
        //qDebug() << "Got Mouse Event";
        emit displayGridClicked((QMouseEvent*) myEvent);
        break;
    }
    }
    return QWidget::event(myEvent);
}
