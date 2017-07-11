#ifndef _BENCHMARK_H_
#define _BENCHMARK_H_

#include "macro.h"

// CPU
#ifdef USE_CPU
// cpu start to time
void cpu_time_tic();

// cpu end to time
void cpu_time_toc();

// return cpu time(ms)
float cpu_time();
#endif



// GPU
#ifdef USE_GPU
// gpu start to time
void gpu_time_tic();

// gpu end to time
void gpu_time_toc();

// return gpu time(ms)
float gpu_time();
#endif

#endif
