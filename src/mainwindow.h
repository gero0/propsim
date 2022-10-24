#pragma once

#include "helpers.h"
#include "qt/clickableLabel.h"
#include "sim.h"

#include <QtWidgets>
#include <memory>

class MainWindow : public QMainWindow {
    Q_OBJECT

public slots:
    void gridClicked(QMouseEvent* e);
    void simToggled(bool checked);
    void pointToggled(bool checked);

public:
    MainWindow();
    ~MainWindow();

private:
    QWidget central_widget { this };
    QHBoxLayout* layout;
    ClickableLabel* image_label;
    QScrollArea* img_scroll;

    QWidget menu_widget { this };
    QVBoxLayout* menu_layout;
    QLabel* data_label;
    QRadioButton* sim_radio;
    QRadioButton* point_radio;
    QPushButton* button;

    Point2D selected_point { 0, 0 };
    Transmitter tx { 0, 0, 23, 2400 };

    std::shared_ptr<Grid> grid;
    std::shared_ptr<QPixmap> pixmap;
    double g_max;
    double g_min;
    double range;
    std::vector<Wall> walls;

    bool point_mode = false;

    float scale_factor = 1.0;

    void launch_sim(uint32_t x = 0, uint32_t y = 0);
    void draw_grid();
    void update_data_label();

    void zoomIn();
    void zoomOut();
    void normalSize();
    void scaleImage(double factor);

    void keyPressEvent(QKeyEvent* event);
};
