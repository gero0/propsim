#pragma once

#include <cstdint>
#include <vector>

enum class PowerUnit {
    dBm,
    mW
};

class Grid {
public:
    Grid(uint32_t size_x, uint32_t size_y);
    double get_val(uint32_t x, uint32_t y, PowerUnit unit = PowerUnit::dBm) const;
    void set_val(uint32_t x, uint32_t y, double val);
    uint32_t size_x;
    uint32_t size_y;
    void print(PowerUnit unit = PowerUnit::dBm);
    double get_max_val(PowerUnit unit = PowerUnit::dBm);
    double get_min_val(PowerUnit unit = PowerUnit::dBm);

private:
    std::vector<double> data;
};

struct Point2D {
    uint32_t x;
    uint32_t y;
};

struct Wall {
    Point2D start;
    Point2D end;
    double attenuation;
};

struct Transmitter {
    Point2D pos;
    double power_dbm;
    double f_MHz;
};

void OSM(Grid& grid, const Transmitter& tx, int n, double scale = 1.0);