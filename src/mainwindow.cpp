#include "mainwindow.h"
#include "sim.h"
#include <memory>
#include <qpainter.h>
#include <qpixmap.h>

#ifdef CUDA_AVAL
#include "cudamatmul.h"
#endif

void MainWindow::launch_sim(uint32_t x, uint32_t y)
{
    const int grid_w = 800;
    const int grid_h = 800;

    if (x >= grid_w || y >= grid_h)
        return;

    grid = std::make_shared<Grid>(grid_w, grid_h);
    Transmitter tx { (int)x, (int)y, 25, 2400 };

    // OSM(*grid, tx, 2);
    std::vector<Wall> walls = std::vector<Wall> {
        { { { 100, 50 }, { 100, 700 } }, 6 },
        { { { 700, 50 }, { 700, 700 } }, 6 },
        { { { 450, 50 }, { 450, 700 } }, 6 },
        { { { 100, 50 }, { 700, 50 } }, 6 },
        { { { 100, 700 }, { 700, 700 } }, 6 },
        { { { 100, 400 }, { 700, 400 } }, 6 },
    };

    MWM(*grid, tx, walls, 2);

    g_max = grid->get_max_val();
    g_min = grid->get_min_val();
    range = g_max - g_min;

    QImage image (grid_w, grid_h, QImage::Format_RGB32);

    for (int x = 0; x < grid_w; x++) {
        for (int y = 0; y < grid_h; y++) {
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

    image_label.setPixmap(*pixmap);
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
#ifdef CUDA_AVAL
    cudaMul();
#endif
}

void MainWindow::gridClicked(QMouseEvent* e) // Implementation of Slot which will consume signal
{
    launch_sim(e->pos().x(), e->pos().y());
}

MainWindow::~MainWindow()
{
    // delete pixmap;
    // delete image_label;
}