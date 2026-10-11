#include <cuda_runtime.h>
#include <cstdio>


__global__ void vecAdd(int* A, int* B, int* C, int N){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if(idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

int main(){

    // declare array A,B and output C
   int N = 10000;
   int *h_A = (*int)malloc(sizeof(int)*N);
   int *h_B = (*int)malloc(sizeof(int)*N);
   int *h_C = (*int)malloc(sizeof(int)*N);

   for(int i =0 ;i<N;i++){
   h_A[i] = i;
   h_B[i] = 2*i;

   }

}

// malloc 2 arrays;
// init arrays on host;
//