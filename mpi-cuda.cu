#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

extern "C" 
{
void runCudaLand( int myrank );
}

__global__ void Hello_kernel( int myrank );

void runCudaLand( int myrank )
{
  printf("MPI rank %d: leaving CPU land \n", myrank );

  cudaSetDevice( myrank % 4 );

  Hello_kernel<<<128,128>>>( myrank );

  printf("MPI rank %d: re-entering CPU land \n", myrank );
}

__global__ void Hello_kernel( int myrank )
{

  int device;

  cudaGetDevice( &device );

  printf("Hello World from CUDA/MPI: Rank %d, Device %d, Thread %d, Block %d \n",
	 myrank, device, threadIdx.x, blockIdx.x );
}
