// thread_grid_2d.cu — 练习 2 参考解答
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void print_coords_2d(int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        if (row % 4 == 0 && col % 4 == 0) {
            printf("block(%d,%d) thread(%d,%d) -> global(%d,%d)\n",
                   blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col);
        }
    }
}

int main() {
    const int rows = 8, cols = 8;

    dim3 block(2, 2);                                  // 每 block 2x2 线程
    dim3 grid((cols + block.x - 1) / block.x,          // 4 个 block/x
              (rows + block.y - 1) / block.y);         // 4 个 block/y
    print_coords_2d<<<grid, block>>>(rows, cols);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("kernel 错误: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
