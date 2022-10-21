#include "sim.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>

Grid::Grid(uint32_t size_x, uint32_t size_y)
    : size_x(size_x)
    , size_y(size_y)
    , data(size_x * size_y)
{
}

double dbm_to_mw(double dbm)
{
    return std::pow(10, dbm / 10.0);
}

double Grid::get_val(uint32_t x, uint32_t y, PowerUnit unit) const
{
    uint32_t index = x * size_x + y;
    if (unit == PowerUnit::dBm)
        return data.at(index);

    return dbm_to_mw(data.at(index));
}

void Grid::set_val(uint32_t x, uint32_t y, double val)
{
    uint32_t index = x * size_x + y;
    data.at(index) = val;
}

void Grid::print(PowerUnit unit)
{
    for (uint32_t x = 0; x < size_x; x++) {
        for (uint32_t y = 0; y < size_y; y++) {
            if (unit == PowerUnit::dBm)
                std::cout << data.at(x * size_x + y) << " ";
            else
                std::cout << dbm_to_mw(data.at(x * size_x + y)) << " ";
        }
        std::cout << "\n";
    }
}

double Grid::get_max_val(PowerUnit unit)
{
    auto it = max_element(std::begin(data), std::end(data));
    if (unit == PowerUnit::dBm)
        return *it;

    return dbm_to_mw(data.at(*it));
}

double Grid::get_min_val(PowerUnit unit)
{
    auto it = min_element(std::begin(data), std::end(data));
    if (unit == PowerUnit::dBm)
        return *it;

    return dbm_to_mw(data.at(*it));
}

void OSM(Grid& grid, const Transmitter& tx, int n, double scale)
{
    for (uint32_t x = 0; x < grid.size_x; x++) {
        for (uint32_t y = 0; y < grid.size_y; y++) {
            if (x == tx.pos.x && y == tx.pos.y) {
                grid.set_val(x, y, tx.power_dbm);
                continue;
            }

            //for now assume TX in 0,0 position
            double distance = std::sqrt(x * x + y * y);
            //calculate L0 for 1m(free space attenuation)
            double L = 20.0 * std::log10(1) + 20.0 * std::log10(tx.f_MHz) - 27.55;

            //add one slope coefficient
            L += 10 * n * log10(distance);

            double val = tx.power_dbm - L;

            grid.set_val(x, y, val);
        }
    }
}