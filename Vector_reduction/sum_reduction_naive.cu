#include <cuda_runtime.h>
#include <iostream>

__global__ void naive_reduction(const float *input, float *output, int N)
{

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

     sdata[tid] = (idx < N) ? input[idx] : 0.0f;
     
    __syncthreads();

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {

        if (tid % (2 * stride) == 0)
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

    int max_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaMalloc(&d_output, max_blocks * sizeof(float));

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    int current_size = N;
    int iteration = 0;

    // Repeated GPU reduction
    while (current_size > 1)
    {

      int num_blocks = (current_size + BLOCK_SIZE - 1) / BLOCK_SIZE;



        naive_reduction<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(float)>>>(d_input, d_output, current_size);

        cudaDeviceSynchronize();

       
        float *temp = d_input;
        d_input = d_output;
        d_output = temp;

        current_size = num_blocks;
    }

    // Copy final result
    float result;

    cudaMemcpy(&result, d_input, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Expected: " << N << std::endl;
    std::cout << "Result: " << result << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);

    delete[] h_input;

    return 0;
}