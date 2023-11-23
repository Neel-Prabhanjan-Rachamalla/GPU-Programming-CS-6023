
/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-3
 * Description: Activation Game 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include "graph.hpp"
 
using namespace std;


ofstream outfile; // The handle for printing the output

/******************************Write your kerenels here ************************************/

//Tranversing level by level.
__global__ void activator(int* d_offset , int* d_csrList , int* d_aid , int* d_apr , int* d_activeVertex , int V , int E , int L , int l , int r , int* r1 , int lev)
{
  int id=l+blockIdx.x*gridDim.y*blockDim.x*blockDim.y+blockIdx.y*blockDim.x*blockDim.y+threadIdx.x*blockDim.y+threadIdx.y;

  //Checking if id is in limit.
  if(id>r) return;

  //Calculating next level's end point atomically.
  for(int i=d_offset[id] ; i<d_offset[id+1] ; i++)
  {
    atomicMax(r1,d_csrList[i]);
  }

  //Cheking if a node is active if aid>=apr and also checking its adjacent neighbours atomically.
  if(d_aid[id]>=d_apr[id] && ((id!=l && id!=r && (d_aid[id+1]>=d_apr[id+1] || d_aid[id-1]>=d_apr[id-1])) || id==l || id==r))
  {
    atomicAdd(&d_activeVertex[lev],1);

    //If the node is active then incrementing the aid of its child nodes atomically.
    for(int i=d_offset[id] ; i<d_offset[id+1] ; i++)
    {
      atomicAdd(&d_aid[d_csrList[i]],1);
    }
  }
}
    
/**************************************END*************************************************/



//Function to write result in output file
void printResult(int *arr, int V,  char* filename){
    outfile.open(filename);
    for(long int i = 0; i < V; i++){
        outfile<<arr[i]<<" ";   
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    int V ; // Number of vertices in the graph
    int E; // Number of edges in the graph
    int L; // number of levels in the graph

    //Reading input graph
    char *inputFilePath = argv[1];
    graph g(inputFilePath);

    //Parsing the graph to create csr list
    g.parseGraph();

    //Reading graph info 
    V = g.num_nodes();
    E = g.num_edges();
    L = g.get_level();


    //Variable for CSR format on host
    int *h_offset; // for csr offset
    int *h_csrList; // for csr
    int *h_apr; // active point requirement

    //reading csr
    h_offset = g.get_offset();
    h_csrList = g.get_csr();   
    h_apr = g.get_aprArray();
    
    // Variables for CSR on device
    int *d_offset;
    int *d_csrList;
    int *d_apr; //activation point requirement array
    int *d_aid; // acive in-degree array
    //Allocating memory on device 
    cudaMalloc(&d_offset, (V+1)*sizeof(int));
    cudaMalloc(&d_csrList, E*sizeof(int)); 
    cudaMalloc(&d_apr, V*sizeof(int)); 
    cudaMalloc(&d_aid, V*sizeof(int));

    //copy the csr offset, csrlist and apr array to device
    cudaMemcpy(d_offset, h_offset, (V+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_csrList, h_csrList, E*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_apr, h_apr, V*sizeof(int), cudaMemcpyHostToDevice);

    // variable for result, storing number of active vertices at each level, on host
    int *h_activeVertex;
    h_activeVertex = (int*)malloc(L*sizeof(int));
    // setting initially all to zero
    memset(h_activeVertex, 0, L*sizeof(int));

    // variable for result, storing number of active vertices at each level, on device
    // int *d_activeVertex;
	// cudaMalloc(&d_activeVertex, L*sizeof(int));


/***Important***/

// Initialize d_aid array to zero for each vertex
// Make sure to use comments

/***END***/
double starttime = rtclock(); 

/*********************************CODE AREA*****************************************/

//Variables declarations required for kernel operations.
int *d_activeVertex,*r1;

//Allocating memory in device.
cudaMalloc(&r1, sizeof(int));
cudaMalloc(&d_activeVertex, L*sizeof(int));

//Initializing the variables in device memory.
cudaMemset(d_aid, 0, V*sizeof(int));
cudaMemset(d_activeVertex, 0, L*sizeof(int));
cudaMemset(r1, 0, sizeof(int));

int l=0,r=0;

//Initializing the first level by looking apr==0.
for(int i=0 ; i<V ; i++)
{
  if(h_apr[i]>0) break;

  r=i;
}

//Traversing through the graph level by level.
for(int i=0 ; i<L ; i++)
{
  dim3 grid(10,1,1);
  dim3 block(1024,1,1);

  activator <<<grid,block>>> (d_offset,d_csrList,d_aid,d_apr,d_activeVertex,V,E,L,l,r,r1,i);
  l=r+1;
  cudaMemcpy(&r,r1,sizeof(int),cudaMemcpyDeviceToHost);
}

//Copying from device to memory the active vertices in a level list.
cudaMemcpy(h_activeVertex,d_activeVertex,L*sizeof(int),cudaMemcpyDeviceToHost);

/********************************END OF CODE AREA**********************************/
double endtime = rtclock();  
printtime("GPU Kernel time: ", starttime, endtime);  

// --> Copy C from Device to Host
char outFIle[30] = "./output.txt" ;
printResult(h_activeVertex, L, outFIle);
if(argc>2)
{
    for(int i=0; i<L; i++)
    {
        printf("level = %d , active nodes = %d\n",i,h_activeVertex[i]);
    }
}

    return 0;
}
