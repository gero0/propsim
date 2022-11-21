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


void hsva_to_rgba(double hsva[4], double rgba[4]) {
    static const double PI = 3.14159265358979323846;
    double h = hsva[0];
    double s = hsva[1] / 255.0;
    double v = hsva[2] / 255.0;
    double a = hsva[3];
    double c = v * s;
    double x = c * (1 - abs(fmod(h / 60.0, 2) - 1));
    double m = v - c;
    double r, g, b;
	if (h < 60) {
		r = c;
		g = x;
		b = 0;
	} else if (h < 120) {
		r = x;
		g = c;
		b = 0;
	} else if (h < 180) {
		r = 0;
		g = c;
		b = x;
	} else if (h < 240) {
		r = 0;
		g = x;
		b = c;
			    
	} else if (h < 300) {
		r = x;
		g = 0;
		b = c;
	} else {
		r = c;
		g = 0;
		b = x;
	}
    rgba[0] = (r + m) * 255.0;
    rgba[1] = (g + m) * 255.0;
    rgba[2] = (b + m) * 255.0;
    rgba[3] = a;
}