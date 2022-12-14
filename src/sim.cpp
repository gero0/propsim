#include "sim.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

bool intersects(Line l1, Line l2);

void OSM(Grid& grid, const Transmitter tx, int n, double scale)
{
    for (int x = 0; x < grid.size_x; x++) {
        for (int y = 0; y < grid.size_y; y++) {

            double distance = std::sqrt(std::pow(x - tx.pos.x, 2) + std::pow(y - tx.pos.y, 2));

            distance *= scale;

            if (distance == 0) {
                grid.set_val(x, y, tx.power_dbm);
                continue;
            }
            //calculate L0 for 1m(free space attenuation)
            double L = 20.0 * std::log10(1 * scale) + 20.0 * std::log10(tx.f_MHz) - 27.55;

            //add one slope coefficient
            L += 10 * n * log10(distance);

            double val = tx.power_dbm - L;

            grid.set_val(x, y, val);
        }
    }
}

void MWM(Grid& grid, const Transmitter tx, std::vector<Wall> walls, int n, double scale)
{
    for (int x = 0; x < grid.size_x; x++) {
        for (int y = 0; y < grid.size_y; y++) {

            double distance = std::sqrt(std::pow(x - tx.pos.x, 2) + std::pow(y - tx.pos.y, 2));

            if (distance < 1) {
                grid.set_val(x, y, tx.power_dbm);
                continue;
            }

            distance *= scale;

            //calculate L0 for 1m(free space attenuation)
            double L = 20.0 * std::log10(1 * scale) + 20.0 * std::log10(tx.f_MHz) - 27.55;

            //add one slope coefficient
            L += 10 * n * log10(distance);

            for (auto wall : walls) {
                Line l { tx.pos.x, tx.pos.y, x, y };
                if (intersects(l, wall.line)) {
                    L += wall.attenuation;
                }
            }

            double val = tx.power_dbm - L;

            grid.set_val(x, y, val);
        }
    }
}

//abandon hope all ye who enter here
bool intersects(Line l1, Line l2)
{
    int X1 = l1.p1.x;
    int Y1 = l1.p1.y;
    int X2 = l1.p2.x;
    int Y2 = l1.p2.y;
    int X3 = l2.p1.x;
    int Y3 = l2.p1.y;
    int X4 = l2.p2.x;
    int Y4 = l2.p2.y;

    Point2D I1 { std::min(X1, X2), std::max(X1, X2) };
    Point2D I2 { std::min(X3, X4), std::max(X3, X4) };

    if (std::max(X1, X2) < std::min(X3, X4)) {
        return false;
    }

    //special case: vertical wall
    if (X3 - X4 == 0) {
        //if both wall and line from tx to point are vertical
        if (X1 - X2 == 0) {
            if (Y1 < std::max(Y3, Y4))
                return (X1 == X3) && (Y2 >= std::max(Y3, Y4));
            else
                return (X1 == X3) && (Y2 <= std::max(Y3, Y4));
        }

        double A1 = (double)(Y1 - Y2) / (X1 - X2);
        double b1 = Y1 - A1 * X1;

        double wY = A1 * X3 + b1;

        if (X3 < I1.x || X3 > I1.y) {
            return false;
        }

        if (Y3 < wY) {
            return Y4 >= wY;
        } else {
            return Y4 <= wY;
        }

        return false;
    }

    //special case for vertical line
    if (X1 - X2 == 0) {
        double A2 = (double)(Y3 - Y4) / (X3 - X4);
        double b2 = Y3 - A2 * X3;

        double wY = A2 * X1 + b2;

        if (X1 < I2.x || X1 > I2.y) {
            return false;
        }

        if (Y1 < wY) {
            return Y2 >= wY;
        } else {
            return Y2 <= wY;
        }

        return false;
    }

    double A1 = (double)(Y1 - Y2) / (X1 - X2);
    double A2 = (double)(Y3 - Y4) / (X3 - X4);

    double b1 = Y1 - A1 * X1;
    double b2 = Y3 - A2 * X3;

    double Xa = (b2 - b1) / (A1 - A2);

    if (
        (Xa <= std::max(std::min(X1, X2), std::min(X3, X4)))
        || (Xa >= std::min(std::max(X1, X2), std::max(X3, X4)))) {
        return false;
    } else
        return true;
}

#ifdef CUDA_AVAL

#include "cuda/mwmkernel.h"

void MWM_CUDA(Grid* grid, const Transmitter tx, std::vector<Wall> walls, int n, double scale)
{
    MWM_CUDA_Wrapper(grid, tx, walls, n, scale);
}
#endif