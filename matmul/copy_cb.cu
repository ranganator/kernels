
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cstdio>
#include <cmath>

constexpr int BM = 256;
constexpr int BN = 128;
constexpr int BK = 8;
constexpr int TM = 16;
constexpr int TN = 8;
constexpr int STAGES = 4;
constexpr int THREADS = 256;
constexpr int N = 8192;

constexpr size_t BYTES = size_t(N) * N * sizeof(float);

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
    // A: 2048 floats, 8 copies per thread.
    #pragma unroll
    for (int p = 0; p < 8; ++p)
    {
        int idx = tid + p * THREADS;

        int row = idx / BK;
        int k = idx % BK;

        int physicalRow = swizzle_a(row, k);

        const float* src =
            A + (blockRow + row) * N + tileK * BK + k;

        float* dst =
            &As[stage][k][physicalRow];

        __pipeline_memcpy_async(dst, src, 4);
    }

    // B: 1024 floats, one 16-byte copy per thread.
    int bk = tid / 32;
    int col = (tid % 32) * 4;

    int physicalCol = swizzle_b(col);

    const float* srcB =
        B + (tileK * BK + bk) * N + blockCol + col;

    float* dstB =
        &Bs[stage][bk][physicalCol];

    __pipeline_memcpy_async(dstB, srcB, 16);

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
    __shared__ __align__(16)
    float As[STAGES][BK][BM];

    __shared__ __align__(16)
    float Bs[STAGES][BK][BN];

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

    // Submit four initial stages.
    #pragma unroll
    for (int t = 0; t < STAGES; ++t)
    {
        issue_tile(
            A, B, As, Bs,
            blockRow, blockCol,
            t, t, tid);
    }

    // Main pipeline.
    for (int tile = 0; tile < numTiles; ++tile)
    {
        int stage = tile % STAGES;

        // Wait for the current stage to finish copying.
        if (tile + STAGES < numTiles)
            __pipeline_wait_prior(3);
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

        // All threads must finish reading this stage
        // before its shared-memory storage is reused.
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

    // Output stores.
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

__global__
void init_matrices(float* A, float* B)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    A[idx] = float((idx % 31) - 15) * 0.01f;
    B[idx] = float((idx % 19) - 9) * 0.02f;
}

int main()
{
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, BYTES);
    cudaMalloc(&d_B, BYTES);
    cudaMalloc(&d_C, BYTES);

    init_matrices<<<(N * N) / 256, 256>>>(d_A, d_B);

    dim3 block(THREADS);
    dim3 grid(N / BN, N / BM);



    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    constexpr int REPEATS = 1;

    cudaEventRecord(start);

    for (int i = 0; i < REPEATS; ++i)
    {
        sgemm_pipelined<<<grid, block>>>(d_A, d_B, d_C);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    float avg_ms = ms / REPEATS;

    double flops = 2.0 * double(N) * N * N;
    double tflops = flops / (avg_ms * 1e9);

    printf("Average time: %.3f ms\n", avg_ms);
    printf("Performance:  %.3f TFLOPS\n", tflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
