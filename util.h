#ifndef _UTIL_H_
#define _UTIL_H_

#include "macro.h"

// handle input
void parseParmeter(int argc, char *argv[], int *n, int *k, char *inputFileName);

// read from file
int readFromFile(char *fileName, DATATYPE* data);

// print array
void printArray(DATATYPE* data, int length);

// zero array
void zeroArray(DATATYPE* data, int length);


#ifdef USE_CPU
// malloc and free on cpu
void* mallocCPUMem(int size);
void freeCPUMem(void *point);
#endif


#ifdef USE_GPU
# include <cuda.h>
# include <cuda_runtime.h>

// malloc and free on cpu
void* mallocGPUMem(int size);
void freeGPUMem(void *point);
// copy from cpu to gpu
void cpu2gpu(void *cpudata, void *gpudata, int size);
// copy from gpu to cpu
void gpu2cpu(void *gpudata, void *cpudata, int size);
// copy from gpu to gpu
void gpu2gpu(void *gpudata_dest, void *gpudata_src, int size);
// cuda error
void handleCudaError(cudaError_t err, const char *file, int line);
#endif

#endif
