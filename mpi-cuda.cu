#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Result from last compute of world.
extern unsigned char *g_resultData;

// Current state of world. 
extern unsigned char *g_data;

// "Above" row
extern unsigned char *g_above_row;

// "Below" row 
extern unsigned char *g_below_row;

static inline void HL_initAllZeros( size_t worldWidth, size_t worldHeight )
{
    size_t g_dataLength = worldWidth * worldHeight;

    // calloc init's to all zeros
    cudaMallocManaged( &g_data, (g_dataLength * sizeof(unsigned char)));
    cudaMallocManaged( &g_resultData, (g_dataLength * sizeof(unsigned char))); 

    size_t i = 0;
    for (i = 0; i < g_dataLength; i++)
    {
        g_data[i] = 0;
        g_resultData[i] = 0;
    }
}

static inline void HL_initReplicator( size_t worldWidth, size_t worldHeight )
{
    HL_initAllZeros(worldWidth, worldHeight);

    size_t x, y;

    x = worldWidth/2;
    y = worldHeight/2;
    
    g_data[x + y*worldWidth + 1] = 1; 
    g_data[x + y*worldWidth + 2] = 1;
    g_data[x + y*worldWidth + 3] = 1;
    g_data[x + (y+1)*worldWidth] = 1;
    g_data[x + (y+2)*worldWidth] = 1;
    g_data[x + (y+3)*worldWidth] = 1; 
}

__global__ void HL_kernel(const unsigned char* d_data, unsigned char* d_resultData,
                            unsigned int worldWidth, unsigned int worldHeight)
{
    size_t index = blockIdx.x *blockDim.x + threadIdx.x;

    for (; index < worldWidth*worldHeight; index += blockDim.x * gridDim.x)
    {
        // get the x and y coords from the index in the flattened world
        size_t y = (size_t) index / worldWidth;

        size_t y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
        size_t y1 = y * worldWidth;
        size_t y2 = ((y + 1) % worldHeight) * worldWidth;

        size_t x1 = index % worldWidth;

        size_t x0 = (x1 + worldWidth - 1) % worldWidth;
        size_t x2 = (x1 + 1) % worldWidth;

        // The rest is similar to the serial code, with the adjacent cells checked
        unsigned int aliveCells = d_data[x0 + y0] + d_data[x1 + y0] + d_data[x2 + y0]
                                    + d_data[x0 + y1] + d_data[x2 + y1]
                                    + d_data[x0 + y2] + d_data[x1 + y2] + d_data[x2 + y2];;

        d_resultData[x1 + y1] = (aliveCells == 3) || (aliveCells == 6 && !d_data[x1 + y1])
          || (aliveCells == 2 && d_data[x1 + y1]) ? 1 : 0;
    }
}

extern "C" void HL_initMaster( unsigned int pattern, size_t worldWidth, size_t worldHeight, int myrank)
{
    // Set device
    int cudaDeviceCount;
    cudaError_t cE; 
    if( (cE = cudaGetDeviceCount( &cudaDeviceCount)) != cudaSuccess )
    {
        printf(" Unable to determine cuda device count, error is %d, count is %d\n", cE, cudaDeviceCount );
        exit(-1);
    }
    if( (cE = cudaSetDevice( myrank % cudaDeviceCount )) != cudaSuccess )
    {
        printf(" Unable to have myrank %d set to cuda device %d, error is %d \n", myrank, (myrank % cudaDeviceCount), cE);
        exit(-1); 
    }

    // INITIALIZE THE CUDA WORLD
    HL_initReplicator( worldWidth, worldHeight );
}


extern "C" void HL_kernelLaunch( unsigned char** d_data, unsigned char** d_resultData, 
        unsigned char * next_above_row, unsigned char * next_below_row, 
        int block_count, int thread_count, 
        unsigned int worldWidth, unsigned int worldHeight, 
        int myrank){
    
    // load back into device
    cudaMemcpy(g_data, next_above_row, worldWidth, cudaMemcpyHostToDevice);
    cudaMemcpy(g_data+(worldHeight- 1) * worldWidth, next_below_row, worldWidth, cudaMemcpyHostToDevice);

    // Call the kernel
    HL_kernel<<<block_count,thread_count>>>(*d_data, *d_resultData, worldWidth, worldHeight);
    cudaDeviceSynchronize();

    //load from device to host
    cudaMemcpy(g_above_row, g_data, worldWidth, cudaMemcpyDeviceToHost);
    cudaMemcpy(g_below_row, g_data+(worldHeight - 1) * worldWidth, worldWidth, cudaMemcpyDeviceToHost);
}


extern "C" void freeCudaArrays(int myrank){
    cudaFree(g_data);
    cudaFree(g_resultData);
}