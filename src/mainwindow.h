#pragma once

#include "qt/clickableLabel.h"
#include "sim.h"
#include <QtWidgets>
#include <memory>
#include <qboxlayout.h>
#include <qevent.h>
#include <qpixmap.h>
#include <qscrollarea.h>

class MainWindow : public QMainWindow {
    Q_OBJECT

public slots:
    void gridClicked(QMouseEvent* e); // Signal to emit

public:
    MainWindow();
    ~MainWindow();

private:
    ClickableLabel image_label;
    QScrollArea img_scroll { this };
    std::shared_ptr<Grid> grid;
    std::shared_ptr<QPixmap> pixmap;
    double g_max;
    double g_min;
    double range;

    void launch_sim(uint32_t x = 100, uint32_t y = 200);
    void draw_grid(const std::vector<Wall>& walls);
    
    void zoomIn();
    void zoomOut();
    void normalSize();
    void scaleImage(double factor);
    void adjustScrollBar(QScrollBar *scrollBar, double factor);

    float scale_factor = 1.0;

    void keyPressEvent(QKeyEvent *event);
};
