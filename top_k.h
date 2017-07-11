#ifndef _TOP_K_H_
#define _TOP_K_H_

#include "macro.h"


#ifdef USE_GPU

// gpu top k
void top_k_gpu(DATATYPE* input, int length, int k, DATATYPE* output);

#endif


#ifdef USE_THRUST
// cuda thrusr top k
void top_k_thrust(DATATYPE* input, int length, int k, DATATYPE* output);
#endif


// cpu top k
void top_k_cpu_serial(DATATYPE* input, int length, int k, DATATYPE* output);


#endif
