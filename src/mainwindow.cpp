#include "mainwindow.h"
#include <chrono>

using namespace std::chrono;

//TODO:
// - Windows support
// - Grid size input
// - Wall input (file or gui?)
// - simulation scale (unit per grid square)
// - presentation - color palette
// - presentation - GPU render
// - runtime CUDA check and error handling
// - another model if we're really bored or sth

void MainWindow::launch_sim(uint32_t x, uint32_t y)
{
    const int grid_w = 1000;
    const int grid_h = 1000;

    if (x >= grid_w || y >= grid_h)
        return;

    tx.pos.x = x;
    tx.pos.y = y;

    grid = std::make_shared<Grid>(grid_w, grid_h);

    auto start = high_resolution_clock::now();

#ifdef CUDA_AVAL
    if (QCoreApplication::arguments().contains("--cpu")) {
        MWM(*grid, tx, walls, 2);
    } else {
        MWM_CUDA(grid.get(), tx, walls, 2);
    }

#else
    if (QCoreApplication::arguments().contains("--cpu")) {
        MWM(*grid, tx, walls, 2);
    } else {
        qDebug() << "CUDA is not supported in this system! Computing using CPU...";
        MWM(*grid, tx, walls, 2);
    }
#endif

    auto stop = high_resolution_clock::now();

    g_max = grid->get_max_val();
    g_min = grid->get_min_val();
    range = g_max - g_min;

    auto draw_start = high_resolution_clock::now();

    draw_grid();

    auto draw_stop = high_resolution_clock::now();

    auto duration_draw = duration_cast<milliseconds>(draw_stop - draw_start);
    auto duration = duration_cast<milliseconds>(stop - start);

    qDebug() << "Simulation (ms): " << duration.count() << "\n"
             << "Drawing (ms): " << duration_draw.count() << "\n";
}

MainWindow::MainWindow()
{
    grid = std::make_shared<Grid>(100, 100);
    pixmap = std::make_shared<QPixmap>(100, 100);
    pixmap->fill(QColor::fromRgb(0, 255, 255));

    layout = new QHBoxLayout();
    image_label = new ClickableLabel();
    img_scroll = new QScrollArea();

    menu_layout = new QVBoxLayout(&menu_widget);
    data_label = new QLabel(&menu_widget);
    sim_radio = new QRadioButton("Place TX on point", &menu_widget);
    point_radio = new QRadioButton("Get data from point", &menu_widget);
    button = new QPushButton("TEST", &menu_widget);

    sim_radio->setChecked(true);
    point_radio->setChecked(false);

    image_label->setPixmap(*pixmap);
    image_label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    image_label->setScaledContents(true);

    img_scroll->setBackgroundRole(QPalette::Dark);
    img_scroll->setWidget(image_label);
    img_scroll->setVisible(true);

    menu_layout->addWidget(data_label);
    menu_layout->addWidget(sim_radio);
    menu_layout->addWidget(point_radio);
    menu_layout->addWidget(button);
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

    connect(image_label, &ClickableLabel::displayGridClicked, this, &MainWindow::gridClicked);
    connect(sim_radio, &QRadioButton::toggled, this, &MainWindow::simToggled);
    connect(point_radio, &QRadioButton::toggled, this, &MainWindow::pointToggled);

    walls = std::vector<Wall> {
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

    launch_sim();
    update_data_label();
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
        selected_point = Point2D { (int)(e->pos().x() / scale_factor), (int)(e->pos().y() / scale_factor) };
        draw_grid();
    } else {
        launch_sim(e->pos().x() / scale_factor, e->pos().y() / scale_factor);
    }

    update_data_label();
}

void MainWindow::draw_grid()
{
    QImage image(grid->size_x, grid->size_y, QImage::Format_RGB32);

    for (int x = 0; x < grid->size_x; x++) {
        for (int y = 0; y < grid->size_y; y++) {
            const int r_val = (((grid->get_val(x, y) - g_min) * 255) / range);
            image.setPixelColor(x, y, QColor::fromRgb(r_val, r_val, r_val));
        }
    }

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