__global__ void multiply_ker(int N, int *C, int *output)
{
    int i=blockDim.y*blockIdx.y+ threadIdx.y;
    int j=blockDim.x*blockIdx.x+ threadIdx.x;
    if(i>=(N>>1) || j>=(N>>1))return;
   
    int sum=0;
    int p=i*2;
    int q=j*2;
    int r=p+1;
    int s=q+1;
    sum+=C[p*N+q]+C[p*N+s]+C[r*N+q]+C[r*N+s];
   
    output[i*(N>>1)+j]=sum;

}
__global__ void mm_ker(int N, int *A, int *B, int *C)
{
    __shared__ int shareA[32][32];
    __shared__ int shareB[32][32];
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int row = by * 32 + ty;
    int col = bx * 32 + tx;
    int temp = 0;
    int width=N;
    for(int i = 0; i < (N+1)/32; ++i){
   if((i*32 + tx)<N and row<N) 
    shareA[ty][tx] = A[row*width + (i*32 + tx)];
   else
    shareA[ty][tx]=0;
   if((i*32 + ty)<N and col<N)
     shareB[ty][tx] = B[(i*32 + ty)*width + col];
   else
     shareB[ty][tx]=0;
   
      __syncthreads();

    for(int k = 0; k < 32; ++k)
     temp += shareA[ty][k] * shareB[k][tx];
    __syncthreads();
            
    }
     if(row<N and col<N)
     C[row*width + col] = temp;
}
void mm(int N, int *matA, int *matB, int *C)
{
// Allocate Unified Memory – accessible from CPU or GPU
    int *d_matA, *d_matB, *d_output;
    cudaMallocManaged(&d_matA, N*N*sizeof(int));
    cudaMallocManaged(&d_matB, N*N*sizeof(int));
    cudaMallocManaged(&d_output, (N)*(N)*sizeof(int));

    dim3 grid_dim(N/32, N/32, 1);
    dim3 block_dim(32,32, 1);

// Copy vectors from host memory to device memory
    cudaMemcpy(d_matA, matA, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, N*N*sizeof(int), cudaMemcpyHostToDevice);

    mm_ker <<<grid_dim, block_dim>>> (N, d_matA, d_matB,  d_output);

// Copy result from device memory to host memory
    cudaMemcpy(C, d_output, (N)*(N)*sizeof(int), cudaMemcpyDeviceToHost);

// Free device memory
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_output);


}
void gputhread(int N, int *matA, int *matB, int *output)
{
// Allocate Unified Memory – accessible from CPU or GPU
    int *d_matA, *d_matB, *d_output,*d_inter;
    cudaMallocManaged(&d_matA, N*N*sizeof(int));
    cudaMallocManaged(&d_matB, N*N*sizeof(int));
    cudaMallocManaged(&d_output, (N>>1)*(N>>1)*sizeof(int));
    cudaMallocManaged(&d_inter, N*N*sizeof(int));
    int *C = new int[(N)*(N)];
    mm(N,matA,matB,C);
    dim3 grid_dim(N/32, N/32, 1);
    dim3 block_dim(32,32, 1);

// Copy vectors from host memory to device memory
    cudaMemcpy(d_matA, matA, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, matB, N*N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inter, C, N*N*sizeof(int), cudaMemcpyHostToDevice);

    multiply_ker <<<grid_dim, block_dim>>> (N, d_inter,  d_output);

// Copy result from device memory to host memory
    cudaMemcpy(output, d_output, (N>>1)*(N>>1)*sizeof(int), cudaMemcpyDeviceToHost);

// Free device memory
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_inter);
    cudaFree(d_output);


}
