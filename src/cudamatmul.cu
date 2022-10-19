#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

__global__ void matMulK(float *A, float *B, float *C, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < width && col < width) {
    float product = 0.0f;
    for (int i = 0; i < width; i++) {
      product += A[row * width + i] * B[i * width + col];
    }
    C[row * width + col] = product;
  }
}

void printMatrix(float *M, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%.2f ", M[i * N + j]);
    }
    printf("\n");
  }
}

void cudaMul(void) {
  // srand(time(NULL));

  const int N = 512;
  float *A, *B, *C;

  cudaMallocManaged(&A, N * N * sizeof(float));
  cudaMallocManaged(&B, N * N * sizeof(float));
  cudaMallocManaged(&C, N * N * sizeof(float));

  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = 1 + i * j;
      B[i * N + j] = 1 + i * j;
    }
  }

  dim3 blocksPerGrid(1, 1);
  dim3 threadsPerBlock(N, N);

  if (N * N > 1024) {
    threadsPerBlock.x = 32;
    threadsPerBlock.y = 32;
    blocksPerGrid.x = (N + threadsPerBlock.x - 1) / threadsPerBlock.x;
    blocksPerGrid.y = (N + threadsPerBlock.y - 1) / threadsPerBlock.y;
  }

  // auto start = std::chrono::high_resolution_clock::now();

  // Run kernel on the gpu
  matMulK<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // auto end = std::chrono::high_resolution_clock::now();

  // auto duration =
  //     std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  printf("%f\n", C[N * N - 1]);
  // std::cout << "CUDA Duration: " << duration.count();

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);
}
