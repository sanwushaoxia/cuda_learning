// thread_grid_2d.cu — 练习 2:二维线程网格与全局索引
//
// 实际应用(矩阵、图像)大多用二维 grid。重点练习:
//   1. dim3 类型与 <<<grid, block>>> 的二维配置
//   2. 二维全局索引: row = blockIdx.y * blockDim.y + threadIdx.y
//                    col = blockIdx.x * blockDim.x + threadIdx.x
//   3. 行主序一维数组与 (row, col) 的换算
//
// 完成后运行: make run T=thread_grid_2d
#include <stdio.h>
#include <cuda_runtime.h>

// TODO: 每个 (row, col) 线程在自己的位置打印一行坐标,
//       格式: "block(%,%) thread(%,%) -> global(%,%)"
//       为了避免刷屏,只在 row % 4 == 0 && col % 4 == 0 时打印
__global__ void print_coords_2d(int rows, int cols) {
    // 在此写代码: 计算 row / col,判断边界,打印
}

int main() {
    const int rows = 8, cols = 8;
    // TODO: 用 dim3 配置 2x2 线程/block(需要 4x4 个 block),
    //       启动 print_coords_2d,然后同步并检查错误
    (void)rows; (void)cols;   // 占位,完成后删除
    printf("thread_grid_2d: 尚未实现\n");
    return 0;
}
