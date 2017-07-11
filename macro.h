#ifndef _MACRO_H_
#define _MACRO_H_

// data type
#define DATATYPE float

// min number
#define NEG_INF -999999999

// use cpu 
#define USE_CPU

// use gpu
#define USE_GPU

#ifdef USE_GPU
    #define HANDLE_CUDA_ERROR( err ) (handleCudaError( err, __FILE__, __LINE__ ))
    #define GPU_BLOCKS_THRESHOLD 2048
    #define GPU_THREADS_THRESHOLD 1024
    #define GPU_SHARED_MEM_THRESHOLD 48*1024
    #define GPU_THREADS 128
#endif
// use gpu thrust
#define USE_THRUST
#endif
