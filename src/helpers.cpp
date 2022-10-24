#include "helpers.h"
#include <algorithm>
#include <cmath>
#include <iostream>

double dbm_to_mw(double dbm)
{
    return std::pow(10, dbm / 10.0);
}

Grid::Grid(uint32_t size_x, uint32_t size_y)
    : size_x(size_x)
    , size_y(size_y)
    , data(size_x * size_y)
{
}

double Grid::get_val(uint32_t x, uint32_t y, PowerUnit unit) const
{
    uint32_t index = y * size_x + x;
    if (unit == PowerUnit::dBm)
        return data.at(index);

    return dbm_to_mw(data.at(index));
}

void Grid::set_val(uint32_t x, uint32_t y, double val)
{
    uint32_t index = y * size_x + x;
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
