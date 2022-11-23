#include "mainwindow.h"
#include <chrono>
#include <exception>
#include <memory>
#include <qdebug.h>
#include <string>
#include <sstream>

#ifdef  CUDA_AVAL
#include "cuda/gridkernel.h"
#endif //  CUDA_AVAL


using namespace std::chrono;


void MainWindow::launch_sim()
{
    grid = std::make_shared<Grid>(grid_w, grid_h);

    auto start = high_resolution_clock::now();

    tx.f_MHz = 2400;
    tx.power_dbm = 20;
    float sim_scale = 0.1;
    try {
        sim_scale = std::stof(scale_input->text().toStdString());
    } catch (std::exception& e) {
        qDebug() << "Could not set scale, using default value: 0.1";
    }

    try {
        tx.f_MHz = std::stoi(txf->text().toStdString());
        tx.power_dbm = std::stoi(txp->text().toStdString());
    } catch (std::exception& e) {
        qDebug() << "Could not set new params for transmitter, using defaults (2.4GHz, 20dBm)";
    }

#ifdef CUDA_AVAL
    if (QCoreApplication::arguments().contains("--cpu")) {
        MWM(*grid, tx, walls, 2, sim_scale);
    } else {
        MWM_CUDA(grid.get(), tx, walls, 2, sim_scale);
    }

#else
    if (QCoreApplication::arguments().contains("--cpu")) {
        MWM(*grid, tx, walls, 2, sim_scale);
    } else {
        qDebug() << "CUDA is not supported in this system! Computing using CPU...";
        MWM(*grid, tx, walls, 2, sim_scale);
    }
#endif

    auto stop = high_resolution_clock::now();

    g_max = grid->get_max_val();
    g_min = grid->get_min_val();
    range = g_max - g_min;

    auto draw_start = high_resolution_clock::now();


//#ifdef CUDA_AVAL
//    if (QCoreApplication::arguments().contains("--cpu")) {
//        draw_grid();
//    } else {
//        //draw_grid_CUDA(); //to implement
//        draw_grid(); //temporary override, to be deleted
//    }
//#else 
//    if (QCoreApplication::arguments().contains("--cpu")) {
//        draw_grid();
//    } else {
//        qDebug() << "CUDA is not supported in this system! Drawing using CPU...";
//        draw_grid();
//    }
//#endif
    draw_grid();

    auto draw_stop = high_resolution_clock::now();

    auto duration_draw = duration_cast<milliseconds>(draw_stop - draw_start);
    auto duration = duration_cast<milliseconds>(stop - start);

    qDebug() << "Simulation (ms): " << duration.count() << "\n"
        << "Drawing (ms): " << duration_draw.count() << "\n";

    update_data_label();
}

MainWindow::MainWindow()
{
    //default grid
    grid = std::make_shared<Grid>(100, 100);
    pixmap = std::make_shared<QPixmap>(100, 100);
    pixmap->fill(QColor::fromRgb(0, 255, 255));

    layout = new QHBoxLayout();
    image_label = new ClickableLabel();
    img_scroll = new QScrollArea();

    menu_layout = new QVBoxLayout(&menu_widget);
    scale_label = new QLabel("Scale: ( 1 point = x m)", &menu_widget);
    scale_input = new QLineEdit("0.1", &menu_widget);
    data_label = new QLabel(&menu_widget);
    sim_radio = new QRadioButton("Place TX on point", &menu_widget);
    point_radio = new QRadioButton("Get data from point", &menu_widget);
    wall_list = new QListWidget(&menu_widget);
    delete_wall_btn = new QPushButton("Delete wall", &menu_widget);
    run_sim_btn = new QPushButton("Run simulation", &menu_widget);

    grid_edit_widget = new QWidget(&menu_widget);
    grid_edit_layout = new QGridLayout(grid_edit_widget);
    grid_w_label = new QLabel("Grid W", grid_edit_widget);
    grid_h_label = new QLabel("Grid H", grid_edit_widget);
    grid_w_input = new QLineEdit("1000", grid_edit_widget);
    grid_h_input = new QLineEdit("1000", grid_edit_widget);

    set_grid_btn = new QPushButton("Resize grid", &menu_widget);

    wall_edit_widget = new QWidget(&menu_widget);
    wall_edit_layout = new QGridLayout(wall_edit_widget);
    wx1l = new QLabel("x1:", wall_edit_widget);
    wy1l = new QLabel("y1:", wall_edit_widget);
    wx2l = new QLabel("x2:", wall_edit_widget);
    wy2l = new QLabel("y2:", wall_edit_widget);
    wLl = new QLabel("Loss (dB): ", wall_edit_widget);
    add_wall_btn = new QPushButton("Add wall", wall_edit_widget);

    wx1 = new QLineEdit(&menu_widget);
    wy1 = new QLineEdit(&menu_widget);
    wx2 = new QLineEdit(&menu_widget);
    wy2 = new QLineEdit(&menu_widget);
    wL = new QLineEdit(&menu_widget);

    tx_edit_widget = new QWidget(&menu_widget);
    tx_edit_layout = new QGridLayout(tx_edit_widget);
    txLabel = new QLabel("Tx settings", tx_edit_widget);
    txfl = new QLabel("f[MHz]:", tx_edit_widget);
    txpl = new QLabel("TX power[dB]:", tx_edit_widget);

    txf = new QLineEdit(tx_edit_widget);
    txp = new QLineEdit(tx_edit_widget);

    txf->setText(QString::number(2400));
    txp->setText(QString::number(20));

    //Grid edit section
    grid_edit_layout->addWidget(grid_w_label, 0, 0);
    grid_edit_layout->addWidget(grid_w_input, 0, 1);
    grid_edit_layout->addWidget(grid_h_label, 0, 2);
    grid_edit_layout->addWidget(grid_h_input, 0, 3);
    grid_edit_widget->setLayout(grid_edit_layout);

    //Wall editor section
    wall_edit_layout->addWidget(wx1l, 0, 0);
    wall_edit_layout->addWidget(wx1, 0, 1);

    wall_edit_layout->addWidget(wy1l, 0, 2);
    wall_edit_layout->addWidget(wy1, 0, 3);

    wall_edit_layout->addWidget(wx2l, 1, 0);
    wall_edit_layout->addWidget(wx2, 1, 1);
    wall_edit_layout->addWidget(wy2l, 1, 2);
    wall_edit_layout->addWidget(wy2, 1, 3);

    wall_edit_layout->addWidget(wLl, 2, 0);
    wall_edit_layout->addWidget(wL, 2, 1);
    wall_edit_layout->addWidget(add_wall_btn, 2, 3);

    wall_edit_widget->setLayout(wall_edit_layout);

    //Tx edit section
    tx_edit_layout->addWidget(txLabel, 0, 0);
    tx_edit_layout->addWidget(txfl, 1, 0);
    tx_edit_layout->addWidget(txf, 1, 1);
    tx_edit_layout->addWidget(txpl, 2, 0);
    tx_edit_layout->addWidget(txp, 2, 1);

    tx_edit_widget->setLayout(tx_edit_layout);

    sim_radio->setChecked(true);
    point_radio->setChecked(false);

    //Map section
    image_label->setPixmap(*pixmap);
    image_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    image_label->setScaledContents(true);

    img_scroll->setBackgroundRole(QPalette::Dark);
    img_scroll->setWidget(image_label);
    img_scroll->setVisible(true);

    //laying out the right part of the window
    menu_layout->addWidget(grid_edit_widget);
    menu_layout->addWidget(set_grid_btn);
    menu_layout->addWidget(scale_label);
    menu_layout->addWidget(scale_input);
    menu_layout->addWidget(data_label);
    menu_layout->addWidget(sim_radio);
    menu_layout->addWidget(point_radio);
    menu_layout->addWidget(wall_edit_widget);
    menu_layout->addWidget(delete_wall_btn);
    menu_layout->addWidget(wall_list);
    menu_layout->addWidget(tx_edit_widget);
    menu_layout->addWidget(run_sim_btn);

    menu_widget.setLayout(menu_layout);

    QSizePolicy spLeft(QSizePolicy::Preferred, QSizePolicy::Preferred);
    spLeft.setHorizontalStretch(3);
    QSizePolicy spRight(QSizePolicy::Preferred, QSizePolicy::Preferred);
    spRight.setHorizontalStretch(1);

    img_scroll->setSizePolicy(spLeft);
    img_scroll->setMinimumSize(400, 400);
    layout->addWidget(img_scroll);

    menu_widget.setSizePolicy(spRight);
    layout->addWidget(&menu_widget);

    central_widget.setLayout(layout);
    setCentralWidget(&central_widget);

    //setting up events
    connect(image_label, &ClickableLabel::displayGridClicked, this, &MainWindow::gridClicked);
    connect(sim_radio, &QRadioButton::toggled, this, &MainWindow::simToggled);
    connect(point_radio, &QRadioButton::toggled, this, &MainWindow::pointToggled);
    connect(delete_wall_btn, &QPushButton::clicked, this, &MainWindow::deleteWall);
    connect(add_wall_btn, &QPushButton::clicked, this, &MainWindow::addWall);
    connect(run_sim_btn, &QPushButton::clicked, this, &MainWindow::launch_sim);
    connect(set_grid_btn, &QPushButton::clicked, this, &MainWindow::setGrid);

    walls = std::vector<Wall>{
        { { { 100, 50 }, { 100, 700 } }, 6 },
        { { { 700, 50 }, { 700, 700 } }, 6 },
        { { { 400, 50 }, { 400, 700 } }, 6 },
        { { { 100, 50 }, { 400, 50 } }, 6 },
        { { { 400, 50 }, { 700, 50 } }, 6 },
        { { { 100, 150 }, { 700, 150 } }, 6 },
        { { { 100, 300 }, { 700, 300 } }, 6 },
        { { { 100, 700 }, { 700, 700 } }, 6 },
        { { { 100, 400 }, { 700, 400 } }, 6 },
    };

    for (auto wall : walls) {
        wall_list->addItem(QString::fromStdString(wallFormat(wall)));
    }

    //draw
    launch_sim();
}

std::string MainWindow::wallFormat(Wall wall)
{
    std::ostringstream out;
    out << "(" << wall.line.p1.x << "," << wall.line.p1.y << ") - "
        << "(" << wall.line.p2.x << "," << wall.line.p2.y << ") L: " << wall.attenuation << " dB";
    return out.str();
}

void MainWindow::setGrid() {
    try {
        grid_w = std::stoi(grid_w_input->text().toStdString());
        grid_h = std::stoi(grid_h_input->text().toStdString());
    } catch (std::exception& e) {
        qDebug() << "Could not set grid size, using default size 1000x1000";
        grid_w = 1000;
        grid_h = 1000;
    }

    grid = std::make_shared<Grid>(grid_w, grid_h);
    launch_sim();
}

void MainWindow::update_data_label()
{
    std::ostringstream out;
    out << "Transmitter position: (" << tx.pos.x << "," << tx.pos.y << ")"
        << "\n"
        << "Power: " << tx.power_dbm << "dBm"
        << "\n"
        << "Value at point (" << selected_point.x << "," << selected_point.x << "): "
        << grid->get_val(selected_point.x, selected_point.y)
        << "dBm \n("
        << (grid->get_val(selected_point.x, selected_point.y, PowerUnit::mW)) << "mW)";

    data_label->setText(QString::fromStdString(out.str()));
}

void MainWindow::addWall()
{
    try {
        Wall w;
        w.line.p1.x = std::stoi(wx1->text().toStdString());
        w.line.p1.y = std::stoi(wy1->text().toStdString());
        w.line.p2.x = std::stoi(wx2->text().toStdString());
        w.line.p2.y = std::stoi(wy2->text().toStdString());
        w.attenuation = std::stoi(wL->text().toStdString());

        walls.push_back(w);
        wall_list->addItem(QString::fromStdString(wallFormat(w)));

        draw_grid();
    } catch (std::exception& e) {
        qDebug() << "Could not add new wall, check if input is correct\n";
    }
}

void MainWindow::deleteWall()
{
    int index = wall_list->currentRow();
    if (index < 0) {
        return;
    }
    walls.erase(walls.begin() + index);
    wall_list->takeItem(index);
    draw_grid();
}

void MainWindow::keyPressEvent(QKeyEvent* event)
{
    qDebug("Screen keyPressEvent %d", event->key());
    switch (event->key()) {
        case Qt::Key_Plus:
            zoomIn();
            break;
        case Qt::Key_Minus:
            zoomOut();
            break;
    }
}

void MainWindow::simToggled(bool checked)
{
    if (checked) {
        point_radio->setChecked(false);
        point_mode = false;
    }
}

void MainWindow::pointToggled(bool checked)
{
    if (checked) {
        sim_radio->setChecked(false);
        point_mode = true;
    }
}

void MainWindow::gridClicked(QMouseEvent* e) // Implementation of Slot which will consume signal
{
    if (point_mode) {
        selected_point = Point2D{ (int)(e->pos().x() / scale_factor), (int)(e->pos().y() / scale_factor) };
        draw_grid();
    } else {
        tx.pos.x = e->pos().x() / scale_factor;
        tx.pos.y = e->pos().y() / scale_factor;
        launch_sim();
    }

    update_data_label();
}

void MainWindow::draw_grid()
{
    QImage image(grid->size_x, grid->size_y, QImage::Format_RGB32);

#ifdef CUDA_AVAL
    if (QCoreApplication::arguments().contains("--cpu")) {
        color_grid_raw(image);
    } else {
        color_grid_raw_cuda(image);// , image);
    }
#else
    if (QCoreApplication::arguments().contains("--cpu")) {
        color_grid(image);
    } else {
        qDebug() << "CUDA is not supported in this system! Drawing using CPU...";
        color_grid(image);
    }
#endif

    pixmap = std::make_shared<QPixmap>(QPixmap::fromImage(image));

    QPainter painter(pixmap.get());
    QPen green((QColor(0, 255, 0)), 1);
    painter.setPen(green);

    for (auto wall : walls) {
        painter.drawLine(wall.line.p1.x, wall.line.p1.y, wall.line.p2.x, wall.line.p2.y);
    }

    painter.drawLine(selected_point.x - 3, selected_point.y - 3, selected_point.x + 3, selected_point.y + 3);
    painter.drawLine(selected_point.x + 3, selected_point.y - 3, selected_point.x - 3, selected_point.y + 3);

    image_label->setPixmap(*pixmap);
    double scale = scale_factor;

    normalSize();
    scaleImage(scale);
}

void MainWindow::color_grid(QImage& image) {
    for (int x = 0; x < grid->size_x; x++) {
        for (int y = 0; y < grid->size_y; y++) {
            //const int r_val = (((grid->get_val(x, y) - g_min) * 255) / range);
            //image.setPixelColor(x, y, QColor::fromRgb(r_val, r_val, r_val));
            const int hue = (((grid->get_val(x, y) - g_min) * 120) / range);
            image.setPixelColor(x, y, QColor::fromHsv(hue, 255, 192, 200));
        }
    }
}

void MainWindow::color_grid_raw(QImage& image) {
    QRgb* raw_image = new QRgb[grid->size_x * grid->size_y];
    for (int x = 0; x < grid->size_x; x++) {
        for (int y = 0; y < grid->size_y; y++) {
            const int hue = (((grid->get_val(x, y) - g_min) * 120) / range);
            double hsva[4];
            hsva[0] = hue;
            hsva[1] = 255;
            hsva[2] = 192;
            hsva[3] = 200;

            double rgba[4];
            hsva_to_rgba(hsva, rgba);
            raw_image[x + y * grid->size_x] = qRgba(rgba[0], rgba[1], rgba[2], rgba[3]);
        }
    }
    for (int x = 0; x < grid->size_x; x++) {
        for (int y = 0; y < grid->size_y; y++) {
            image.setPixel(x, y, raw_image[x + y * grid->size_x]);
        }
    }
	delete[] raw_image;
}
#ifdef CUDA_AVAL
void MainWindow::color_grid_raw_cuda(QImage& image) {
    unsigned int* raw_image = new QRgb[grid->data.size()];
    grid_CUDA_Wrapper(grid.get(), raw_image);
    for (int x = 0; x < grid->size_x; x++) {
        for (int y = 0; y < grid->size_y; y++) {
            image.setPixel(x, y, raw_image[x + y * grid->size_x]);
        }
    }
    delete[] raw_image;
}
#endif
void MainWindow::zoomIn()
{
    scaleImage(1.25);
}

void MainWindow::zoomOut()
{
    scaleImage(0.8);
}

void MainWindow::normalSize()
{
    image_label->adjustSize();
    scale_factor = 1.0;
}

void MainWindow::scaleImage(double factor)
{
    scale_factor *= factor;
    image_label->resize(scale_factor * image_label->pixmap(Qt::ReturnByValue).size());
}

MainWindow::~MainWindow()
{
}