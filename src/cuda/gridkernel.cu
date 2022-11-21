#include "gridkernel.h"
#include <iostream>

__device__ void hsva_to_rgba_CUDA(double hsva[4], double rgba[4]);

__global__ void grid_kernel(double* grid, int grid_w, int grid_h, double g_min, double g_max, unsigned int* qrgb) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < grid_w && y < grid_h) {
        const int hue = ((grid[x + y * grid_w] - g_min) * 120) / (g_max - g_min);
        double hsva[4];
        hsva[0] = hue;
        hsva[1] = 255;
        hsva[2] = 192;
        hsva[3] = 200;
        
        double rgba[4];
        hsva_to_rgba_CUDA(hsva, rgba);
        qrgb[x + y * grid_w] = ((int(rgba[3]) & 0xffu) << 24) | ((int(rgba[0]) & 0xffu) << 16) | ((int(rgba[1]) & 0xffu) << 8) | (int(rgba[2]) & 0xffu);
    }
}

void grid_CUDA_Wrapper(Grid* grid, unsigned int* raw_image){//, QImage image) {
    double* g;
    unsigned int* qrgb;
    //unsigned int* raw_image = new unsigned int[grid->data.size()];
    cudaMalloc(&g, grid->data.size() * sizeof(double));
    //cudaMalloc(&raw_image, grid->data.size() * sizeof(unsigned int));
    cudaMalloc(&qrgb, grid->data.size() * sizeof(unsigned int));

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(1, 1);
    blocksPerGrid.x = (grid->size_x + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (grid->size_y + threadsPerBlock.y - 1) / threadsPerBlock.y;

    cudaMemcpy(g, grid->data.data(), grid->data.size() * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(qrgb, raw_image, grid->data.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    //cudaMemcpy(raw, raw_image, grid->data.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);

    grid_kernel << <blocksPerGrid, threadsPerBlock >> > (g, grid->size_x, grid->size_y, grid->get_min_val(), grid->get_max_val(), qrgb);

    cudaDeviceSynchronize();

    cudaMemcpy(grid->data.data(), g, grid->data.size() * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(raw_image, qrgb, grid->data.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    //cudaMemcpy(raw_image, raw, grid->data.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(g);
    cudaFree(qrgb);
    
    //for (int x = 0; x < grid->size_x; x++)
    //    for (int y = 0; y < grid->size_y; y++)
    //        image.setPixel(x, y, raw_image[x + y * grid->size_x]);
    
    //delete[] raw_image;

}

__device__ void hsva_to_rgba_CUDA(double hsva[4], double rgba[4]) {
    double h = hsva[0];
    double s = hsva[1] / 255.0;
    double v = hsva[2] / 255.0;
    double a = hsva[3];
    double c = v * s;
    double x = c * (1 - abs(fmodf(h / 60.0, 2) - 1));
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