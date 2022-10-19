#include <iostream>
#ifdef CUDA_AVAL
#include "cudamatmul.h"
#endif

int main(void)
{
#ifdef CUDA_AVAL
    cudaMul();
#endif
    std::cout << "Hello world";
    return 0;
}