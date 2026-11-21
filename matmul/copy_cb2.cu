
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

constexpr int N = 8192;

constexpr int BM = 256;
constexpr int BN = 128;
constexpr int BK = 16;
constexpr int TM = 16;
constexpr int TN = 8;

constexpr int STAGES = 2;  // Test 2, 3, 4
constexpr int THREADS = 256;

constexpr size_t BYTES = size_t(N) * N * sizeof(float);

#define CUDA_CHECK(call)                                       \
    do {                                                       \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess) {                              \
            printf("CUDA error: %s\n", cudaGetErrorString(err));\
            std::exit(EXIT_FAILURE);                           \
        }                                                      \
    } while (0)

__device__ __forceinline__
int swizzle_a(int row, int k)
{
    return row ^ ((row & 32) >> 3) ^ (k << 2);
}

__device__ __forceinline__
int swizzle_b(int col)
{
    return col ^ ((col & 32) >> 3);
}

__device__ __forceinline__
void issue_tile(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float (&As)[STAGES][BK][BM],
    float (&Bs)[STAGES][BK][BN],
    int blockRow,
    int blockCol,
    int tileK,
    int stage,
    int tid)
{
    #pragma unroll
    for (int p = 0; p < 16; ++p)
    {
        int idx = tid + p * THREADS;

        int row = idx / BK;
        int k = idx % BK;

        int physicalRow = swizzle_a(row, k);

        const float* src =
            A + (blockRow + row) * N + tileK * BK + k;

        float* dst = &As[stage][k][physicalRow];

        __pipeline_memcpy_async(dst, src, 4);
    }

#pragma unroll
for (int p = 0; p < 2; ++p)
{
    int idx = tid + p * THREADS;

    int bk = idx / (BN / 4);
    int col = (idx % (BN / 4)) * 4;

    int physicalCol = swizzle_b(col);

    const float* src =
        B + (tileK * BK + bk) * N + blockCol + col;

    float* dst = &Bs[stage][bk][physicalCol];

    __pipeline_memcpy_async(dst, src, 16);
}

    __pipeline_commit();
}

__device__ __forceinline__
void load_fragment(
    const float (&As)[STAGES][BK][BM],
    const float (&Bs)[STAGES][BK][BN],
    int stage,
    int k,
    int threadRow,
    int threadCol,
    float (&a)[TM],
    float (&b)[TN])
{
    #pragma unroll
    for (int q = 0; q < 4; ++q)
    {
        int row = threadRow + q * 4;
        int physicalRow = swizzle_a(row, k);

        float4 v = *reinterpret_cast<const float4*>(
            &As[stage][k][physicalRow]);

        a[q * 4 + 0] = v.x;
        a[q * 4 + 1] = v.y;
        a[q * 4 + 2] = v.z;
        a[q * 4 + 3] = v.w;
    }

    #pragma unroll
    for (int q = 0; q < 2; ++q)
    {
        int col = threadCol + q * 4;
        int physicalCol = swizzle_b(col);

        float4 v = *reinterpret_cast<const float4*>(
            &Bs[stage][k][physicalCol]);

        b[q * 4 + 0] = v.x;
        b[q * 4 + 1] = v.y;
        b[q * 4 + 2] = v.z;
        b[q * 4 + 3] = v.w;
    }
}

__global__ __launch_bounds__(THREADS)
void sgemm_pipelined(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C)
{
    __shared__ __align__(16) float As[STAGES][BK][BM];
    __shared__ __align__(16) float Bs[STAGES][BK][BN];

    int tid = threadIdx.x;
    int warpId = tid >> 5;
    int laneId = tid & 31;

    int warpRow = warpId / 2;
    int warpCol = warpId % 2;

    int threadRow =
        warpRow * 64 + (laneId / 8) * TM;

    int threadCol =
        warpCol * 64 + (laneId % 8) * TN;

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    float acc[TM][TN] = {};

    constexpr int numTiles = N / BK;

    // Initial pipeline stages
    #pragma unroll
    for (int t = 0; t < STAGES; ++t)
    {
        issue_tile(
            A, B, As, Bs,
            blockRow, blockCol,
            t, t, tid);
    }

    // Main compute loop
    for (int tile = 0; tile < numTiles; ++tile)
    {
        int stage = tile % STAGES;

        // Supports STAGES = 2, 3, or 4
        if (tile + STAGES < numTiles)
            __pipeline_wait_prior(STAGES - 1);
        else
            __pipeline_wait_prior(0);

        __syncthreads();

        float aReg[2][TM];
        float bReg[2][TN];

        load_fragment(
            As, Bs, stage, 0,
            threadRow, threadCol,
            aReg[0], bReg[0]);

        #pragma unroll
        for (int k = 0; k < BK; ++k)
        {
            int curr = k & 1;
            int next = curr ^ 1;

            if (k + 1 < BK)
            {
                load_fragment(
                    As, Bs, stage, k + 1,
                    threadRow, threadCol,
                    aReg[next], bReg[next]);
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i)
            {
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                {
                    acc[i][j] = fmaf(
                        aReg[curr][i],
                        bReg[curr][j],
                        acc[i][j]);
                }
            }
        }

        // Finish reading the current shared-memory stage
        __syncthreads();

        int future = tile + STAGES;

        if (future < numTiles)
        {
            issue_tile(
                A, B, As, Bs,
                blockRow, blockCol,
                future, stage, tid);
        }
    }

    // Output
    #pragma unroll
    for (int i = 0; i < TM; ++i)
    {
        int row = blockRow + threadRow + i;
        int col = blockCol + threadCol;

        float4 out0 = make_float4(
            acc[i][0], acc[i][1],
            acc[i][2], acc[i][3]);

        float4 out1 = make_float4(
            acc[i][4], acc[i][5],
            acc[i][6], acc[i][7]);

        float4* dst = reinterpret_cast<float4*>(
            C + row * N + col);

        dst[0] = out0;
        dst[1] = out1;
    }
}

int main()
{
    float *d_A, *d_B, *d_C;

    CUDA_CHECK(cudaMalloc(&d_A, BYTES));
    CUDA_CHECK(cudaMalloc(&d_B, BYTES));
    CUDA_CHECK(cudaMalloc(&d_C, BYTES));

    // Initialize on CPU: no extra CUDA kernel launches.
    std::vector<float> h_A(size_t(N) * N);
    std::vector<float> h_B(size_t(N) * N);

    for (size_t i = 0; i < size_t(N) * N; ++i)
    {
        h_A[i] = float(int(i % 31) - 15) * 0.01f;
        h_B[i] = float(int(i % 19) - 9) * 0.02f;
    }

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A.data(), BYTES, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B.data(), BYTES, cudaMemcpyHostToDevice));

    dim3 block(THREADS);
    dim3 grid(N / BN, N / BM);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    // EXACTLY ONE GEMM KERNEL LAUNCH
    sgemm_pipelined<<<grid, block>>>(d_A, d_B, d_C);

    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    double flops = 2.0 * double(N) * N * N;
    double tflops = flops / (ms * 1e9);

    printf("Pipeline stages: %d\n", STAGES);
    printf("Time: %.3f ms\n", ms);
    printf("Performance: %.3f TFLOPS\n", tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
