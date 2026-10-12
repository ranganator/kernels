#include <iostream>
#include <vector>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void tiled_matmul(const float *A, const float *B, float *C, int M, int K, int N)
{

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int NUM_TILES = (K + TILE_SIZE - 1) / TILE_SIZE;

    float sum = 0.0f;
    for (int t = 0; t < NUM_TILES; t++)
    {
        if (row < M && (TILE_SIZE * t + threadIdx.x) < K)
        {

            sA[threadIdx.y][threadIdx.x] = A[row * K + (TILE_SIZE * t + threadIdx.x)];
        }
        else
        {

            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if ( (t * TILE_SIZE + threadIdx.y) < K && col < N)
        {
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        }
        else
        {

            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // each tile
        for (int i = 0; i < TILE_SIZE; i++)
        {

            sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
    {
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
    for (auto &x : h_A) x = (float)rand() / RAND_MAX;
    for (auto &x : h_B) x = (float)rand() / RAND_MAX;

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, bytesA);
    cudaMalloc(&d_B, bytesB);
    cudaMalloc(&d_C, bytesC );

    cudaMemcpy(d_A, h_A.data(), bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytesB, cudaMemcpyHostToDevice);

    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y);

    tiled_matmul<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

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

    return 0;
}