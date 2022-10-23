#include "mainwindow.h"
#include "sim.h"
#include <memory>
#include <qdebug.h>
#include <qnamespace.h>
#include <qpainter.h>
#include <qpixmap.h>

#include <chrono>
#include <qsize.h>
#include <qsizepolicy.h>
#include <vector>
using namespace std::chrono;

void MainWindow::launch_sim(uint32_t x, uint32_t y)
{
    const int grid_w = 1000;
    const int grid_h = 1000;

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

    auto draw_start = high_resolution_clock::now();

    draw_grid(walls);

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

    image_label.setPixmap(*pixmap);
    image_label.setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    image_label.setScaledContents(true);

    img_scroll.setBackgroundRole(QPalette::Dark);
    img_scroll.setWidget(&image_label);
    img_scroll.setVisible(true);

    setCentralWidget(&img_scroll);

    connect(&image_label, &ClickableLabel::displayGridClicked, this, &MainWindow::gridClicked);

    launch_sim();
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

void MainWindow::draw_grid(const std::vector<Wall>& walls)
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

    image_label.setPixmap(*pixmap);
    double scale = scale_factor;

    normalSize();
    scaleImage(scale);
}

void MainWindow::gridClicked(QMouseEvent* e) // Implementation of Slot which will consume signal
{
    launch_sim(e->pos().x() / scale_factor, e->pos().y() / scale_factor);
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
    image_label.adjustSize();
    scale_factor = 1.0;
}

void MainWindow::scaleImage(double factor)
{
    scale_factor *= factor;
    image_label.resize(scale_factor * image_label.pixmap(Qt::ReturnByValue).size());

    adjustScrollBar(img_scroll.horizontalScrollBar(), factor);
    adjustScrollBar(img_scroll.verticalScrollBar(), factor);
}

void MainWindow::adjustScrollBar(QScrollBar* scrollBar, double factor)
{
    scrollBar->setValue(int(factor * scrollBar->value()
        + ((factor - 1) * scrollBar->pageStep() / 2)));
}

MainWindow::~MainWindow()
{
}