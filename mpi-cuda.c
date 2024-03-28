#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

unsigned char *g_resultData=NULL;

unsigned char *g_data=NULL;

unsigned char *g_above_row=NULL;

unsigned char *g_below_row=NULL;

extern void HL_initMaster(int pattern, size_t worldWidth, size_t worldHeight, int myrank );

// ----- IMPORT KERNEL FUNCTION ----- //
void HL_kernelLaunch( unsigned char** d_data, unsigned char** d_resultData, 
        unsigned char* next_above_row, unsigned char* next_below_row, 
        int block_count, int thread_count, 
        unsigned int worldWidth, unsigned int worldHeight, 
        int myrank);

// ----- IMPORT FREE MEMORY FUNCTION ----- //
extern void freeCudaArrays(int myrank);

// ----- SWAP POINTER FUNCTIONS ----- //
static inline void HL_swapPointers( unsigned char **pA, unsigned char **pB)
{
    // Create temporary holder to hold A's values
    unsigned char *temporary;
    temporary = *pA;

    // Perform the swap
    *pA = *pB;
    *pB = temporary;
}

static inline void HL_printWorld(size_t iteration)
{
    int i, j;

    printf("Print World - Iteration %lu \n", iteration);
    
    for( i = 0; i < g_worldHeight; i++)
    {
	printf("Row %2d: ", i);
	for( j = 0; j < g_worldWidth; j++)
	{
	    printf("%u ", (unsigned int)g_data[(i*g_worldWidth) + j]);
	}
	printf("\n");
    }

    printf("\n\n");
}

// extern void runCudaLand( int myrank );

int main(int argc, char** argv) {
	unsigned int pattern = 0;
	unsigned int worldSize = 0;
	unsigned int iterations = 0;
	unsigned int thread_count = 0;

	pattern = atoi(argv[1]);
	worldSize = atoi(argv[2]);
	iterations = atoi(argv[3]);
	thread_count = atoi(argv[4]);

	//create local storage for top and bottom rows
	g_above_row = calloc( worldSize,sizeof(unsigned char));
	g_below_row = calloc( worldSize,sizeof(unsigned char));
	unsigned char * next_above_row = calloc( worldSize,sizeof(unsigned char));
	unsigned char * next_below_row = calloc( worldSize,sizeof(unsigned char));

	// setput MPI
	MPI_Init(&argc, &argv);

	// Get the number of processes
	int numranks;
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);

	// Get the rank of the process
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	int block_count = (worldSize * worldSize) / thread_count;

	HL_initMaster(pattern,worldSize,iterations, myrank);

	double t0, t1;
	if(myrank == 0){
		t0 = MPI_Wtime();
	}

	int aboveRank = ( numranks + myrank - 1 ) % numranks ;
	int belowRank = ( numranks + myrank + 1 ) % numranks ;

	int tag1 = 10;
	int tag2 = 20;

	MPI_Status stat;
	MPI_Request send_request_above, recv_request_above, send_request_below, recv_request_below;

	for(int tick = 0; tick < iterations; ++tick) {
		// storage for recieves

		// recieve requests for above and below rows
		MPI_Irecv(next_below_row, worldSize, MPI_CHAR, aboveRank, tag1, MPI_COMM_WORLD, &recv_request_below);
		MPI_Irecv(next_above_row, worldSize, MPI_CHAR, belowRank, tag2, MPI_COMM_WORLD, &recv_request_above);

		// sends for above and below rows
		MPI_Isend(g_below_row, worldSize, MPI_CHAR, belowRank, tag1, MPI_COMM_WORLD, &send_request_below);		
		MPI_Isend(g_above_row, worldSize, MPI_CHAR, aboveRank, tag2, MPI_COMM_WORLD, &send_request_above);

		// wait 
		MPI_Wait(&send_request_below, &stat);
		MPI_Wait(&send_request_above, &stat);
		MPI_Wait(&recv_request_below, &stat);
		MPI_Wait(&recv_request_above, &stat);

		// Call the Kernel function
		HL_kernelLaunch(&g_data, &g_resultData, next_above_row, next_below_row, block_count, thread_count, worldSize, worldSize, myrank);

		// Swap the global data
		HL_swapPointers(&g_data, &g_resultData);
  	}

  	MPI_Barrier( MPI_COMM_WORLD );

	if(myrank == 0){
		t1 = MPI_Wtime();
		printf("\n[WORLD SIZE] %d x %d\n[ITERATIONS] %d\n[EXECUTION TIME] %f\n", worldSize, worldSize, iterations, t1-t0);
	}

	HL_printWorld(iterations);

	// free cuda memory
	freeCudaArrays(myrank);
	free(g_above_row);
	free(g_below_row);
	free(next_above_row);
	free(next_below_row);

	// Finalize the MPI environment.
	MPI_Finalize();
}
