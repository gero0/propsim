#include "mainwindow.h"
#include "sim.h"
#include <memory>
#include <qdebug.h>
#include <qpainter.h>
#include <qpixmap.h>

#include <chrono>
using namespace std::chrono;

void MainWindow::launch_sim(uint32_t x, uint32_t y)
{
    const int grid_w = 2000;
    const int grid_h = 2000;

    if (x >= grid_w || y >= grid_h)
        return;

    grid = std::make_shared<Grid>(grid_w, grid_h);
    Transmitter tx { (int)x, (int)y, 25, 2400 };

    // OSM(*grid, tx, 2);
    std::vector<Wall> walls = std::vector<Wall> {
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

    QImage image(grid_w, grid_h, QImage::Format_RGB32);

    auto draw_start = high_resolution_clock::now();

    for (int x = 0; x < grid_w; x++) {
        for (int y = 0; y < grid_h; y++) {
            const int r_val = (((grid->get_val(x, y) - g_min) * 255) / range);
            image.setPixelColor(x, y, QColor::fromRgb(r_val, r_val, r_val));
        }
    }

    auto draw_stop = high_resolution_clock::now();

    pixmap = std::make_shared<QPixmap>(QPixmap::fromImage(image));

    QPainter painter(pixmap.get());
    QPen green((QColor(0, 255, 0)), 1);
    painter.setPen(green);

    for (auto wall : walls) {
        painter.drawLine(wall.line.p1.x, wall.line.p1.y, wall.line.p2.x, wall.line.p2.y);
    }

    image_label.setPixmap(*pixmap);

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

    image_label.setPixmap(*pixmap);
    image_label.show();

    setCentralWidget(&image_label);

    connect(&image_label, &ClickableLabel::displayGridClicked, this, &MainWindow::gridClicked);

    launch_sim();
}

void MainWindow::gridClicked(QMouseEvent* e) // Implementation of Slot which will consume signal
{
    launch_sim(e->pos().x(), e->pos().y());
}

MainWindow::~MainWindow()
{
}