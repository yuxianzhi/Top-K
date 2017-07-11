#include <stdlib.h>

#include "benchmark.h"
#include "macro.h"

// CPU
#ifdef USE_CPU
#include "sys/time.h"

struct timeval cpu_start, cpu_end;

// cpu start to time
void cpu_time_tic()
{
    gettimeofday(&cpu_start, NULL);
}

// cpu end to time
void cpu_time_toc()
{
    gettimeofday(&cpu_end, NULL);
}

// return cpu time(ms)
float cpu_time()
{
    float time_elapsed=(cpu_end.tv_sec-cpu_start.tv_sec)*1000.0 + (cpu_end.tv_usec-cpu_start.tv_usec)/1000.0;
    return time_elapsed;
}
#endif




// GPU
#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cuda.h>

cudaEvent_t gpu_start, gpu_end;

// gpu start to time
void gpu_time_tic()
{
    cudaEventCreate(&gpu_start);
    cudaEventRecord(gpu_start, 0);
}

// gpu end to time
void gpu_time_toc()
{
    cudaEventCreate(&gpu_end);
    cudaEventRecord(gpu_end, 0);
}

// return gpu time(ms)
float gpu_time()
{
    float time_elapsed=0;
    cudaEventSynchronize(gpu_start); 
    cudaEventSynchronize(gpu_end); 
    cudaEventElapsedTime(&time_elapsed, gpu_start, gpu_end); 
    cudaEventDestroy(gpu_start); 
    cudaEventDestroy(gpu_end);
}
#endif
