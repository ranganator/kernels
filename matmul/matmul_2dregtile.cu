#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <iomanip>

constexpr int BM = 64;

constexpr int BK = 8;

constexpr int BN = 64;

constexpr int TM = 4;

constexpr int TN = 4;

__global__ void matmul_2dreg(const float *A, const float *B, float *C, int M, int K, int N)
{

__shared__ float As[BM][BK];
__shared__ float Bs[BK][BN];


int blockRow = blockDim.y * BM; // this is to know which output indices should be in the current block, because each block outputs 64x64 elements

int blockCol = blockDim.x * BN; 

int threadRow = threadIdx.y * TM; // because each thread computes a TM x TN output elements, we compute threadrow and threadcol to find out what elements the thread
                                  // should work on

int threadCol = threadIdx.x * TN;

int tid = threadIdx.y * blockDim.x  + threadIdx.x;// flat thread index

int num_threads = blockDim.x*blockDim.y;

float acc[TM][TN] = {0.0f}; // the 4x4 output elements which is being accumulated into from multiplying registrs

float Areg[TM]; // values to load into registers from As

float Breg[TN]; // values to load into registers from Bs
// why did we choose our block size?
for(int k0 = 0; k0<K; k0+= BK ){
// here we load a 64x8 tile from A into As and a 8x64 tile into Bs each iteration

// As moves horizontally, Bs moves vertically, so the col in As is controlled by k0 and the row in Bs is controlled by K0;

for( int idx = tid; idx < BM*BK; idx += num_threads ){ // since the number of threads is less than the number of elements in the shared block, each thread loads 2 elements
int Arow = tid/BK;

int Acol = tid % BK;

int globalArow = blockRow + Arow;

int globalAcol = Acol + k0;

if(globalArow < M && globalAcol < K){

    As[Arow][Acol] = A[globalArow*K + globalAcol];
}else{

    As[Arow][Acol] = 0.0f;
}
}

for( int idx = tid; tid < BK*BN; idx += num_threads ){ // since the number of threads is less than the number of elements in the shared block, each thread loads 2 elements
int Brow = tid/BN;

int Bcol = tid % BN;

int globalBrow = Brow + k0;

int globalBcol = blockCol + Bcol;

if(globalBrow < K && globalBcol < N){

    Bs[Brow][Bcol] = A[globalBrow*N + globalBcol];
}else{

    Bs[Brow][Bcol] = 0.0f;
}
}

__syncthreads(); // wait for threads to load everything into tiles from global memory


for(int k = 0; k<BK; k++){ // after we load each tile, within each tile we go through each slice

 // now we have to load from shared memory into registers 

// so how this works is that one thread is loading the first element of the first TM rows into Areg and is loading the first element of the first TN cols into B reg

for(int i = 0; i<TM;i++){

    Areg[i] = As[threadRow+i][k];
}

for(int j = 0; j<TN;j++){

    Breg[j] = Bs[k][threadCol+j];
}


for(int i = 0; i <TM;i++){

    for(int j = 0; j < TN;j++){

        acc[i][j] += Areg[i]*Breg[j];
    }
}  

}

}

__syncthreads();



int globalCrow = blockRow + threadRow;

int globalCcol = blockCol + threadCol;

for(int i = 0; i<TM;i++){

    for(int j = 0; j< TN; j++){
         
        if(globalCrow+ i < M && globalCcol + j < N){
        C[(globalCrow+i)*N  + globalCcol] = acc[i][j];
        }else{

           C[(globalCrow+i)*N  + globalCcol] = 0.0f;  

    }
}
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
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    matmul_2dreg<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

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