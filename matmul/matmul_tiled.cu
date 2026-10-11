#include <iostream>
#include <vector>
#include <cuda_runtime.h>


__global__ void naive_matmul(const float* A,const float* B,float* C,int N){

    int row =  blockDim.y*blockIdx.y + threadIdx.y;
    int col =  blockDim.x*blockIdx.x + threadIdx.x;
    float sum = 0.0f;
    if(row < N && col < N){
    for(int i = 0; i<N;i++){
     sum += A[row*N +i] *B[i*N + col];
    }
    C[row*N + col] = sum;
}
}

// int cpu_matmul(const float* A,const float* B,float* C,int N){


// }
int main() {
    // 1. Matrix dimensions
  
    const int K = 1024;
    const int M = 1024;
    const int N = 1024;
    const size_t bytes = N * N * sizeof(float);

    // 2. Allocate and initialize matrices on CPU
    std::vector<float> h_A (N * N, 1.0f);
    std::vector<float> h_B(N * N, 2.0f);
    std::vector<float> h_C(N * N, 0.0f);


    float *d_A, *d_B, *d_C;

    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    
    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    
    dim3 block(16, 16);

    dim3 grid(
        (N + block.x - 1) / block.x,
        (N + block.y - 1) / block.y
    );

    
    naive_matmul<<<grid, block>>>(d_A, d_B, d_C, N);

    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << cudaGetErrorString(err) << "\n";
        return 1;
    }

    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

  

   
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}