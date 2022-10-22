#pragma once

#include <QtWidgets>
#include <qboxlayout.h>
#include <memory>
#include <qevent.h>
#include <qpixmap.h>
#include "sim.h"
#include "clickableLabel.h"

class MainWindow : public QMainWindow {
    Q_OBJECT

public slots:
    void gridClicked(QMouseEvent* e); // Signal to emit

public:
    MainWindow();
   ~MainWindow();

private:
    ClickableLabel image_label {this};

    std::shared_ptr<Grid> grid;
    std::shared_ptr<QPixmap> pixmap;
    double g_max;
    double g_min;
    double range;

    void launch_sim(uint32_t x = 100, uint32_t y = 200);
};
