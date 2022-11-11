#pragma once

#include "helpers.h"
#include <cstdint>
#include <vector>

void OSM(Grid& grid, const Transmitter tx, int n, double scale = 1.0);
void MWM(Grid& grid, const Transmitter tx, std::vector<Wall> walls, int n, double scale = 0.1);

#ifdef CUDA_AVAL
void MWM_CUDA(Grid* grid, const Transmitter tx, std::vector<Wall> walls, int n, double scale = 0.1);
#endif