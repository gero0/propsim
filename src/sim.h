#pragma once

#include <cstdint>
#include <vector>
#include "helpers.h"

void OSM(Grid& grid, const Transmitter tx, int n, double scale = 1.0);
void MWM(Grid& grid, const Transmitter tx, std::vector<Wall> walls, int n, double scale = 1.0);

#ifdef CUDA_AVAL
void MWM_CUDA(Grid* grid, const Transmitter tx, std::vector<Wall> walls, int n, double scale = 1.0);
#endif