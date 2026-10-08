#include <stdio.h>

// __global__: 声明设备端 kernel，由 CPU 调用、在 GPU 上执行，返回类型必须为 void
__global__ void cuda_say_hello(){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello world, CUDA! %d, %d\n", threadIdx.x, i);
}

int main(){
    printf("Hello world, CPU\n");
    cuda_say_hello<<<3,2>>>();

    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    return 0;
}