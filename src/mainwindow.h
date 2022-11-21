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
    void deleteWall();
    void addWall();

public:
    MainWindow();
    ~MainWindow();

private:
    QWidget central_widget { this };
    QHBoxLayout* layout;
    ClickableLabel* image_label;
    QScrollArea* img_scroll;

    QWidget menu_widget { this };

    QWidget* grid_edit_widget;
    QGridLayout* grid_edit_layout;
    QLabel* grid_w_label;
    QLineEdit* grid_w_input;
    QLabel* grid_h_label;
    QLineEdit* grid_h_input;
    QPushButton* set_grid_btn;

    QVBoxLayout* menu_layout;
    QLabel* scale_label;
    QLineEdit* scale_input;
    QLabel* data_label;
    QRadioButton* sim_radio;
    QRadioButton* point_radio;
    QListWidget* wall_list;
    QPushButton* delete_wall_btn;
    QPushButton* run_sim_btn;

    QWidget* wall_edit_widget;
    QGridLayout* wall_edit_layout;
    QLabel* wx1l;
    QLabel* wy1l;
    QLabel* wx2l;
    QLabel* wy2l;
    QLabel* wLl;
    QPushButton* add_wall_btn;

    QWidget* tx_edit_widget;
    QGridLayout* tx_edit_layout;
    QLineEdit* wx1;
    QLineEdit* wy1;
    QLineEdit* wx2;
    QLineEdit* wy2;
    QLineEdit* wL;

    QLabel* txLabel;
    QLabel* txfl;
    QLabel* txpl;
    QLineEdit* txf;
    QLineEdit* txp;

    Point2D selected_point { 0, 0 };
    Transmitter tx { 0, 0, 20, 2400 };

    std::shared_ptr<Grid> grid;
    std::shared_ptr<QPixmap> pixmap;
    double g_max;
    double g_min;
    double range;
    std::vector<Wall> walls;

    bool point_mode = false;

    float scale_factor = 1.0;
    int grid_w = 1000;
    int grid_h = 1000;

    void launch_sim();
    void draw_grid();
    void color_grid(QImage& image);
    void color_grid_raw(QImage& image);
    void update_data_label();

    void zoomIn();
    void zoomOut();
    void normalSize();
    void scaleImage(double factor);
    void setGrid();

    void keyPressEvent(QKeyEvent* event);

    std::string wallFormat(Wall wall);
};
