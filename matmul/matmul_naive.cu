#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>

__global__ void naive_matmul(const float *A, const float *B, float *C, int M, int K, int N)
{

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    float sum = 0.0f;
    if (row < N && col < N)
    {
        for (int i = 0; i < N; i++)
        {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

void cpu_matmul(const float *A, const float *B, float *C, int M, int K, int N)
{

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void print_rows(const char *label, const float *C, int N, int rows, int cols)
{
    std::cout << label << ":\n";
    for (int i = 0; i < rows; i++)
    {
        std::cout << "Row " << i << ": ";
        for (int j = 0; j < cols; j++)
        {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main()
{
    // 1. Matrix dimensions

    const int M = 1024;
    const int K = 1024;
    const int N = 1024;

    const size_t bytesA = (size_t)M * K * sizeof(float);
    const size_t bytesB = (size_t)K * N * sizeof(float);
    const size_t bytesC = (size_t)M * N * sizeof(float);

    // 2. Allocate and initialize matrices on CPU
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_gpu(M * N, 0.0f);
    std::vector<float> h_C_cpu(M * N, 0.0f);

    srand(42);
    for (auto &x : h_A)
        x = (float)rand() / RAND_MAX;
    for (auto &x : h_B)
        x = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC);

    cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice);

    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y);

    naive_matmul<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << cudaGetErrorString(err) << "\n";
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_C_gpu.data(), d_C, bytesC, cudaMemcpyDeviceToHost);

    cpu_matmul(h_A.data(), h_B.data(), h_C_cpu.data(), M, K, N);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    std::cout << std::fixed << std::setprecision(4);
    const int PRINT_ROWS = 4;
    const int PRINT_COLS = N; // change to e.g. 8 to see only the start of each row

    print_rows("CPU result", h_C_cpu.data(), N, PRINT_ROWS, PRINT_COLS);
    print_rows("GPU result", h_C_gpu.data(), N, PRINT_ROWS, PRINT_COLS);
    return 0;
}