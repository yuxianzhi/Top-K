#include <stdio.h>
#include <stdlib.h>

#include "util.h"
#include "benchmark.h"
#include "top_k.h"
#include "macro.h"

#ifdef USE_GPU
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif

int main(int argc, char *argv[])
{
    char *inputFileName;
    int k,n;
    parseParmeter(argc, argv, &n, &k, inputFileName);
    
    DATATYPE *data, *output;
    int data_size=sizeof(DATATYPE)*n;
    int output_size=sizeof(DATATYPE)*k;
    data = (DATATYPE *)mallocCPUMem(data_size);
    output = (DATATYPE *)mallocCPUMem(output_size);
    int nn = readFromFile(argv[1], data);
    if( nn != n)
         printf("too much data in file(%d %d)\n", nn, n);
    cpu_time_tic();
    top_k_cpu_serial(data, n, k, output);
    cpu_time_toc();
    printf("CPU Result %f ms:\n", cpu_time());
    printArray(output, k);

#ifdef USE_GPU
    DATATYPE *data_D, *output_D;
    data_D = (DATATYPE *)mallocGPUMem(data_size);
    output_D = (DATATYPE *)mallocGPUMem(output_size); 
    cpu2gpu(data, data_D, data_size);
    gpu_time_tic();
    top_k_gpu(data_D, n, k, output_D);
    gpu_time_toc();
    zeroArray(output, k);
    gpu2cpu(output_D, output, output_size);
    printf("GPU Result %f ms:\n", gpu_time());
    printArray(output, k);
    freeGPUMem(data_D);
    freeGPUMem(output_D);
#endif

#ifdef USE_THRUST
    zeroArray(output, k);
    cpu_time_tic();
    top_k_thrust(data, n, k, output);
    cpu_time_toc();
    printf("Thrust Result %f ms:\n", cpu_time());
    printArray(output, k);
#endif
    freeCPUMem(data);
    freeCPUMem(output);
    return 0;
}
