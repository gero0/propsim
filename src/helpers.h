#pragma once

#include<cstdint>
#include<vector>

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
    std::vector<double> data;
};

struct Point2D {
    int x;
    int y;
};

struct Line {
    Point2D p1;
    Point2D p2;
};

struct Wall {
    Line line;
    double attenuation;
};

struct Transmitter {
    Point2D pos;
    double power_dbm;
    double f_MHz;
};

struct SimResults {
    Grid g;
    double g_max;
    double g_min;
    double range;
};
