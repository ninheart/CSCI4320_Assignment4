#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>
#include<unistd.h>
#include<stdbool.h>

unsigned char *g_resultData=NULL;

unsigned char *g_data=NULL;

unsigned char *g_aboveRow=NULL;

unsigned char *g_belowRow=NULL;

extern void HL_initMaster(int pattern, size_t worldWidth, size_t worldHeight, int myrank );

// ----- IMPORT KERNEL FUNCTION ----- //
extern void HL_kernelLaunch( unsigned char** d_data, unsigned char** d_resultData, unsigned char** d_aboveRow, unsigned char** d_belowRow, int block_count, int thread_count, unsigned int worldWidth, unsigned int worldHeight, int myrank );

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


// extern void runCudaLand( int myrank );

int main(int argc, char** argv) {


	unsigned int pattern = 0;
	unsigned int worldSize = 0;
	unsigned int iterations = 0;
	unsigned int thread_count = 0;

	MPI_Status stat;
	MPI_Request send_request, recv_request;
	MPI_Init(&argc, &argv);

	// Get the number of processes
	int numranks;
	MPI_Comm_size(MPI_COMM_WORLD, &numranks);

	// Get the rank of the process
	int myrank;
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);


	pattern = atoi(argv[1]);
	worldSize = atoi(argv[2]);
	iterations = atoi(argv[3]);
	thread_count = atoi(argv[4]);

	int block_count = (worldSize * worldSize) / thread_count;

	HL_initMaster(pattern,worldSize,iterations,thread_count);

	double t0, t1;
	if(myrank == 0){
		t0 = MPI_Wtime();
	}

	int aboveRank = ( numranks + myrank - 1 ) % numranks ;
	int belowRank = ( numranks + myrank + 1 ) % numranks ;

	int tag1 = 10;
	int tag2 = 20;

	MPI_Barrier( MPI_COMM_WORLD );

	for(int tick = 0; tick < iterations; ++tick) {
		MPI_Irecv(g_belowRow, worldSize, MPI_UNSIGNED_CHAR, aboveRank, tag1, MPI_COMM_WORLD, &recv_request);
		MPI_Isend(g_belowRow, worldSize, MPI_UNSIGNED_CHAR, belowRank, tag1, MPI_COMM_WORLD, &send_request);

		// Barrier
		MPI_Wait(&send_request, &stat);
		MPI_Wait(&recv_request, &stat);

		// Send and receive to below row
		MPI_Irecv(g_aboveRow, worldSize, MPI_UNSIGNED_CHAR, belowRank, tag2, MPI_COMM_WORLD, &recv_request);
		MPI_Isend(g_aboveRow, worldSize, MPI_UNSIGNED_CHAR, aboveRank, tag2, MPI_COMM_WORLD, &send_request);

		// Barrier
		MPI_Wait(&send_request, &stat);
		MPI_Wait(&recv_request, &stat);

		// SWAP THE POINTERS
		HL_swapPointers(&g_aboveRow, &g_belowRow);
		MPI_Barrier( MPI_COMM_WORLD );

		// Call the Kernel function
		HL_kernelLaunch(&g_data, &g_resultData, &g_aboveRow, &g_belowRow, block_count, thread_count, worldSize, worldSize, myrank);

		// Swap the global data
		HL_swapPointers(&g_data, &g_resultData);

		// Barrier
		MPI_Barrier( MPI_COMM_WORLD );
  	}

	if(myrank == 0){
		t1 = MPI_Wtime();
		printf("\n[WORLD SIZE] %d x %d\n[ITERATIONS] %d\n[EXECUTION TIME] %f\n", worldSize, worldSize, iterations, t1-t0);
	}

	freeCudaArrays(myrank);

	MPI_Barrier( MPI_COMM_WORLD );

  // Finalize the MPI environment.
  MPI_Finalize();
}
