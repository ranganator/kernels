#include <cuda_runtime.h>
#include <iostream>

__global__ void naive_reduction(const float *input, float *output, int N)
{

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockDim.x*2) + tid;

    float a = (idx < N) ? input[idx] : 0.0f;
    float b = (idx + blockDim.x < N) ? input[idx+blockDim.x] : 0.0f;

    sdata[tid] = a+b;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        output[blockIdx.x] = sdata[0];
    }
}

int main()
{
    constexpr int N = 1 << 22; // 4,194,304
    constexpr int BLOCK_SIZE = 256;

    float *h_input = new float[N];

    for (int i = 0; i < N; i++)
    {
        h_input[i] = 1.0f;
    }

    float *d_input, *d_output;

    cudaMalloc(&d_input, N * sizeof(float));

    int max_blocks = (N + 2*BLOCK_SIZE - 1) / (2*BLOCK_SIZE);
    cudaMalloc(&d_output, max_blocks * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int current_size = N;
    int iteration = 0;

    while (current_size > 1)
    {

        int num_blocks = (current_size + 2*BLOCK_SIZE - 1) / (2*BLOCK_SIZE);

        naive_reduction<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, current_size);

        cudaDeviceSynchronize();

        float *temp = d_input;
        d_input = d_output;
        d_output = temp;

        current_size = num_blocks;
    }

    float result;

    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Expected: " << N << std::endl;
    std::cout << "Result: " << result << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;

    return 0;
}