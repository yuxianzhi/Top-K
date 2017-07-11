#include "top_k.h"
#include "macro.h"

#ifdef USE_THRUST
#include <cuda_runtime.h>
#include <cuda.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>

void top_k_thrust(DATATYPE* input, int length, int k, DATATYPE* output)
{
    thrust::sort(thrust::host, input, input+length, thrust::greater<DATATYPE>());
    memcpy(output, input, sizeof(DATATYPE)*k);
}

#endif
