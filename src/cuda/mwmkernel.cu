#include "mwmkernel.h"
#include <iostream>

__device__ bool cuda_intersects(Line l1, Line l2);

__global__ void MWM_kernel(double *grid, int grid_w, int grid_h,
                           const Transmitter tx, Wall *walls, int wall_count,
                           int n, double scale) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < grid_w && y < grid_h) {

    double distance = sqrt(pow(x - tx.pos.x, 2) + pow(y - tx.pos.y, 2));
    if (distance == 0) {
      grid[x * grid_w + y] = tx.power_dbm;
      return;
    }

    double L = 20.0 * log10f(1) + 20.0 * log10f(tx.f_MHz) - 27.55;
    L += 10 * n * log10f(distance);

    for (int i = 0; i < wall_count; i++) {
      Line l{tx.pos.x, tx.pos.y, x, y};
      if (cuda_intersects(l, walls[i].line)) {
        L += walls[i].attenuation;
      }
    }

    grid[x * grid_w +y] = tx.power_dbm - L;
    // grid[x * grid_w + y] = blockIdx.x + blockIdx.y;
  }
}

void MWM_CUDA_Wrapper(Grid* grid, const Transmitter tx, std::vector<Wall> walls,
                      int n, double scale) {
  double *g;
  Wall *w;
  cudaMalloc(&g, grid->data.size() * sizeof(double));
  cudaMalloc(&w, walls.size() * sizeof(Wall));

  dim3 threadsPerBlock(32, 32);

  dim3 blocksPerGrid(1, 1);
  blocksPerGrid.x = (grid->size_x + threadsPerBlock.x - 1) / threadsPerBlock.x;
  blocksPerGrid.y = (grid->size_y + threadsPerBlock.y - 1) / threadsPerBlock.y;

  cudaMemcpy(g, grid->data.data(), grid->data.size() * sizeof(double),
             cudaMemcpyHostToDevice);

  cudaMemcpy(w, walls.data(), walls.size() * sizeof(Wall),
             cudaMemcpyHostToDevice);

  MWM_kernel<<<blocksPerGrid, threadsPerBlock>>>(g, grid->size_x, grid->size_y,
                                                 tx, w, walls.size(), n, scale);

  cudaDeviceSynchronize();

  cudaMemcpy(grid->data.data(), g, grid->data.size() * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(g);
  cudaFree(w);
}

__device__ bool cuda_intersects(Line l1, Line l2) {
  int X1 = l1.p1.x;
  int Y1 = l1.p1.y;
  int X2 = l1.p2.x;
  int Y2 = l1.p2.y;
  int X3 = l2.p1.x;
  int Y3 = l2.p1.y;
  int X4 = l2.p2.x;
  int Y4 = l2.p2.y;

  Point2D I1{min(X1, X2), max(X1, X2)};
  Point2D I2{min(X3, X4), max(X3, X4)};

  if (max(X1, X2) < min(X3, X4)) {
    return false;
  }

  // special case: vertical wall
  if (X3 - X4 == 0) {
    // if both wall and line from tx to point are vertical
    if (X1 - X2 == 0) {
      if (Y1 < max(Y3, Y4))
        return (X1 == X3) && (Y2 >= max(Y3, Y4));
      else
        return (X1 == X3) && (Y2 <= max(Y3, Y4));
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

  // special case for vertical line
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

  if ((Xa <= max(min(X1, X2), min(X3, X4))) ||
      (Xa >= min(max(X1, X2), max(X3, X4)))) {
    return false;
  } else
    return true;
}
