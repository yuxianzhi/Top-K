#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "macro.h"


// handle input
void parseParmeter(int argc, char *argv[], int *n, int *k, char *inputFileName){
    if(argc < 4 )
    {
        printf("input must be 3 parameters, such as <./main filename n k>\n");
        exit(1);
    }

    inputFileName = argv[1];
    *n = atoi(argv[2]);
    *k = atoi(argv[3]);
#ifdef DEBUG
    printf("select %d from %d\n", *k, *n);
#endif
}


// read from file
int readFromFile(char *fileName, DATATYPE* data)
{
    FILE *fp;  
    if((fp=fopen(fileName, "r")) == NULL) {  
        printf("file %s cannot be opened/n", fileName);  
        exit(1);  
    }
  
    int i=0;
    while(!feof(fp)) {  
        fscanf(fp, "%f ", &data[i]);
        i++;
    }
    
    fclose(fp); 
    return i; 
}


// print array
void printArray(DATATYPE* data, int length)
{
    for(int i=0; i<length; i++)
        printf("%f ", data[i]);
    printf("\n");
}

// zero array
void zeroArray(DATATYPE* data, int length)
{
    for(int i=0; i<length; i++)
        data[i] = 0;
}


#ifdef USE_CPU
// malloc and free on cpu
void* mallocCPUMem(int size)
{
    if(size>0)
        return malloc(size);
    else
        return NULL;
}

void freeCPUMem(void *point)
{
    if(point != NULL)
        free(point);
}
#endif


#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cuda.h>

// malloc and free on cpu
void* mallocGPUMem(int size)
{
    if(size<=0)
        return NULL;

    void* data;
    HANDLE_CUDA_ERROR(cudaMalloc(&data, size));
    return data;
}

void freeGPUMem(void *point)
{
    if(point != NULL)
        HANDLE_CUDA_ERROR(cudaFree(point));
}

// copy from cpu to gpu
void cpu2gpu(void *cpudata, void *gpudata, int size)
{
    if(size<=0)
        return;

    HANDLE_CUDA_ERROR(cudaMemcpy(gpudata, cpudata, size, cudaMemcpyHostToDevice));
}

// copy from gpu to cpu
void gpu2cpu(void *gpudata, void *cpudata, int size)
{
    if(size<=0)
        return;

    HANDLE_CUDA_ERROR(cudaMemcpy(cpudata, gpudata, size, cudaMemcpyDeviceToHost));
}

// copy from gpu to gpu
void gpu2gpu(void *gpudata_dest, void *gpudata_src, int size)
{
    if(size<=0)
        return;

    HANDLE_CUDA_ERROR(cudaMemcpy(gpudata_dest, gpudata_src, size, cudaMemcpyDeviceToDevice));
}


// cuda error
void handleCudaError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ), file,line );
        exit(2);
    }
}
#endif
