
#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

#define WARP_SIZE 32

__global__ void transpose(int* d_matrixD , int* d_matrixDT , int r , int q)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;

    if (i<r && j<q)
    {
      d_matrixDT[j*r+i]=d_matrixD[i*q+j];
    }
}

__global__ void multiplication(int *A , int *B , int *C , int *DT , int *E , int p , int q , int r) {
    __shared__ float shared_A[WARP_SIZE*WARP_SIZE];
    __shared__ float shared_B[WARP_SIZE*WARP_SIZE];
    __shared__ float shared_C[WARP_SIZE*WARP_SIZE];
    __shared__ float shared_DT[WARP_SIZE*WARP_SIZE];

    int id1=threadIdx.x;
		int id2=threadIdx.y;

    int row=blockIdx.y*WARP_SIZE+id2;
    int col=blockIdx.x*WARP_SIZE+id1;

		int sum=0;

    for (int i=0 ; i<=q/WARP_SIZE ; i++)
		{
        if(row<p && i*WARP_SIZE+id1<q)
				{
            shared_A[id2*WARP_SIZE+id1]=A[row*q+i*WARP_SIZE+id1];
            shared_C[id2*WARP_SIZE+id1]=C[row*q+i*WARP_SIZE+id1];
        }
				else
				{
            shared_A[id2*WARP_SIZE+id1] = 0;
            shared_C[id2*WARP_SIZE+id1] = 0;
        }

        if (i*WARP_SIZE+id2<q && col<r)
				{
            shared_B[id2*WARP_SIZE+id1]=B[(i*WARP_SIZE+id2)*r+col];
            shared_DT[id2*WARP_SIZE+id1]=DT[(i*WARP_SIZE+id2)*r+col];
        }
				else
				{
            shared_B[id2*WARP_SIZE+id1] = 0;
            shared_DT[id2*WARP_SIZE+id1] = 0;
        }

        __syncthreads();

        for(int i=0 ; i<WARP_SIZE ; i++)
				{
            sum+=shared_A[id2*WARP_SIZE+i]*shared_B[i*WARP_SIZE+id1];
        }

        for (int i=0 ; i<WARP_SIZE ; i++)
				{
            sum+=shared_C[id2*WARP_SIZE+i]*shared_DT[i*WARP_SIZE+id1];
        }

        __syncthreads();
    }

    if(row<p && col<r)
		{
        E[row*r+col]=sum;
    }
}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE;
	
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	int *d_matrixDT;
	cudaMalloc(&d_matrixDT, q * r * sizeof(int));

  dim3 grid1(r/WARP_SIZE+1,q/WARP_SIZE+1);
	dim3 block1(WARP_SIZE,WARP_SIZE);

  transpose <<<grid1,block1>>> (d_matrixD,d_matrixDT,r,q);

  dim3 grid2((r-1)/WARP_SIZE+1,(p-1)/WARP_SIZE+1);
	dim3 block2(WARP_SIZE,WARP_SIZE);

  multiplication <<<grid2,block2>>> (d_matrixA,d_matrixB,d_matrixC,d_matrixDT,d_matrixE,p,q,r);

	cudaDeviceSynchronize();

	cudaFree(d_matrixDT);

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
