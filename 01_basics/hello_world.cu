#include <stdio.h>

// __global__: 声明设备端 kernel，由 CPU 调用、在 GPU 上执行，返回类型必须为 void
__global__ void cuda_say_hello(){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello world, CUDA! %d, %d\n", threadIdx.x, i);
}

int main(){
    printf("Hello world, CPU\n");
    cuda_say_hello<<<3,2>>>();

    // kernel 启动是异步的:cudaDeviceSynchronize() 等 kernel 真正跑完,
    // GPU 的 printf 才保证已输出;返回值同时兜住「执行期」错误。
    // 注意它只覆盖一半:启动瞬间的「配置类」错误(如 block 超 1024)
    // 存在待取错误槽里,要 cudaGetLastError() 才能取到(取走即清空,
    // 同步就查不到了)。两阶段齐全 + 出错即退的写法见根目录 cuda_common.cuh
    // 的 CUDA_CHECK_LAUNCH();每个 kernel 启动后都应调用它。
    cudaError_t cudaerr = cudaDeviceSynchronize();
    if (cudaerr != cudaSuccess)
        printf("kernel launch failed with error \"%s\".\n",
               cudaGetErrorString(cudaerr));
    return 0;
}