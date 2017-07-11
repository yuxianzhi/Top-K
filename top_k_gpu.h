#ifndef _TOP_K_GPU_H_
#define _TOP_K_GPU_H_

#include "macro.h"


#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cuda.h>

// gpu top k
void top_k_gpu(DATATYPE* input, int length, int k, DATATYPE* output);

#endif

#endif
