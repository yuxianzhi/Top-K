#include "top_k.h"
#include "util.h"
#include "macro.h"
#include <stdio.h>

#ifdef USE_GPU
#include <cuda_runtime.h>
#include <cuda.h>



__device__ inline void replace_smaller(DATATYPE* array, int k, DATATYPE data)
{
    if(data < array[k-1])
        return;
    for(int j=k-2; j>=0; j--)
    {
        if(data > array[j])
            array[j+1] = array[j];
        else{
            array[j+1] = data;
            return;
        }
    }
    array[0] = data;
}


__global__ void top_k_gpu_kernel1(DATATYPE* input, int length, int k, DATATYPE* output)
{
    // produce k data in decent order
    output[0] = input[0];
    for(int i=1; i<k; i++)
    {
        output[i] = NEG_INF;
	replace_smaller(output, i+1, input[i]);
    }
    
    // replace the data if bigger
    for(int i=k; i<length; i++)
    {
	replace_smaller(output, k, input[i]);
    }
}

__global__ void top_k_gpu_kernel2(DATATYPE* input, int length, int k, DATATYPE* output)
{
    extern __shared__ DATATYPE shared_buffer[];
    DATATYPE *myPoint = shared_buffer + threadIdx.x * k;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadNum = gridDim.x * blockDim.x;
    int i, index;
    
    for(index=0,i=threadId; index<k; index++,i+=threadNum)
    {
        myPoint[index] = NEG_INF;
        replace_smaller(myPoint, index+1, input[i]);
    }
    // replace the data if bigger
    for(int i=k*threadNum+threadId; i<length; i+=threadNum)
    {
        replace_smaller(myPoint, k, input[i]);
    }
    __syncthreads();

    if(threadIdx.x == 0)
    {
	// produce k data in decent order
        for(i=0; i<k; i++)
        {
            output[i] = myPoint[i];
        }
    
        // replace the data if bigger
        for(i=k; i<k*threadNum; i++)
        {
            replace_smaller(output, k, myPoint[i]);
        }
    }
}

__device__ inline void mergeTwoK(DATATYPE* left, DATATYPE* right, int k)
{
    int i;
    for(i=0; i<k; i++)
    {
        replace_smaller(left, k, right[i]);
    }
}
__global__ void top_k_gpu_kernel3_1(DATATYPE* input, int length, int k, DATATYPE* output)
{
    extern __shared__ DATATYPE shared_buffer[];
    DATATYPE *myPoint = shared_buffer + threadIdx.x * k;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadNum = gridDim.x * blockDim.x;
    int localThreadId = threadIdx.x;
    int localThreadNum = blockDim.x;
    int i, index;

    for(index=0,i=blockIdx.x*localThreadNum*k+localThreadId; index<k; index++,i+=localThreadNum)
    {
        myPoint[index] = NEG_INF;
        replace_smaller(myPoint, index+1, input[i]);
    }
    // replace the data if bigger
    for(i=k*threadNum+threadId; i<length; i+=threadNum)
    {
        replace_smaller(myPoint, k, input[i]);
    }
    __syncthreads();


    // reduction
    for(i=localThreadNum>>1; i>0; i>>=1) {
        if(localThreadId < i)
        {
	    mergeTwoK(myPoint, myPoint+i*k, k);
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        // produce k data in decent order
        index = blockIdx.x*localThreadNum*k;
        for(i=0; i<k; i++)
        {
            input[index+i] = myPoint[i];
        }
    }
}
__global__ void top_k_gpu_kernel3_2(DATATYPE* input, int num, int stride, int k, DATATYPE* output)
{
    extern __shared__ DATATYPE shared_buffer[];
    DATATYPE *myPoint = shared_buffer + threadIdx.x * k;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadNum = gridDim.x * blockDim.x;
    int localThreadId = threadIdx.x;
    int localThreadNum = blockDim.x;
    int i;

    for(i=0; i<k; i++)
 	myPoint[i] = input[threadId*stride + i];
    for(i=threadNum+threadId; i<num; i+=threadNum)
        mergeTwoK(myPoint, input+i*stride, k);
    __syncthreads();

    // reduction
    for(i=localThreadNum>>1; i>0; i>>=1) {
        if(localThreadId < i)
        {   
            mergeTwoK(myPoint, myPoint+i*k, k);
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        // produce k data in decent order
        DATATYPE *outputPoint = output + blockIdx.x*localThreadNum*stride;
        for(i=0; i<k; i++)
        {
            outputPoint[i] = myPoint[i];
        }
    }
}


__global__ void top_k_gpu_kernel3_1_orig(DATATYPE* input, int length, int k, DATATYPE* output)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadNum = gridDim.x * blockDim.x;
    int localThreadId = threadIdx.x;
    int localThreadNum = blockDim.x;
    DATATYPE *myPoint = input + threadId*k;
    int i, index;

    for(index=0; index<k; index++)
    {
        replace_smaller(myPoint, index+1, myPoint[index]);
    }
    // replace the data if bigger
    for(i=k*threadNum+threadId; i<length; i+=threadNum)
    {
        replace_smaller(myPoint, k, input[i]);
    }
    __syncthreads();


    // reduction
    for(i=localThreadNum>>1; i>0; i>>=1) {
        if(localThreadId < i)
        {
	    mergeTwoK(myPoint, myPoint+i*k, k);
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        // produce k data in decent order
        DATATYPE *outputPoint = input + blockIdx.x*localThreadNum*k;
        for(i=0; i<k; i++)
        {
            outputPoint[i] = myPoint[i];
        }
    }
}
__global__ void top_k_gpu_kernel3_2_orig(DATATYPE* input, int num, int stride, int k, DATATYPE* output)
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadNum = gridDim.x * blockDim.x;
    int localThreadId = threadIdx.x;
    int localThreadNum = blockDim.x;
    int i;
    DATATYPE *myPoint = input + threadId*stride;

    for(i=threadNum+threadId; i<num; i+=threadNum)
        mergeTwoK(myPoint, input+i*stride, k);
    __syncthreads();

    // reduction
    for(i=localThreadNum>>1; i>0; i>>=1) {
        if(localThreadId < i)
        {   
            mergeTwoK(myPoint, myPoint+i*stride, k);
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        // produce k data in decent order
        DATATYPE *outputPoint = output + blockIdx.x*localThreadNum*stride;
        for(i=0; i<k; i++)
        {
            outputPoint[i] = myPoint[i];
        }
    }
}



__device__ inline void replace_smaller_stride(DATATYPE* array, int k, int stride, DATATYPE data)
{
    DATATYPE *prev, *next;
    prev = array + (k-1)*stride;
    if(data < *prev)
        return;
    for(int j=k-2; j>=0; j--)
    {
	next = prev;
        prev = prev - stride;
        if(data > *prev)
            *next = *prev;
        else{
            *next = data;
            return;
        }
    }
    *array = data;
}
__device__ inline void mergeTwoK_stride(DATATYPE* left, DATATYPE* right, int k, int stride)
{
    int i;
    DATATYPE* current = right;
    for(i=0; i<k; i++,current+=stride)
    {
        replace_smaller_stride(left, k, stride, *current);
    }
}
__device__ inline void mergeTwoK_stride_lin(DATATYPE* left, DATATYPE* right, int k, int stride)
{
    int i;
    DATATYPE* current = right;
    for(i=0; i<k; i++,current++)
    {
        replace_smaller_stride(left, k, stride, *current);
    }
}

__global__ void top_k_gpu_kernel4_1(DATATYPE* input, int length, int k, DATATYPE* output)
{
    extern __shared__ DATATYPE shared_buffer[];
    DATATYPE *myPoint = shared_buffer + threadIdx.x;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadNum = gridDim.x * blockDim.x;
    int localThreadId = threadIdx.x;
    int localThreadNum = blockDim.x;
    int stride = localThreadNum; 
    int i, index;

    DATATYPE *current = myPoint;
    DATATYPE *global_point = input + blockIdx.x*localThreadNum*k+localThreadId;
    for(index=0; index<k; index++,current+=stride,global_point+=localThreadNum)
    {
        *current = NEG_INF;
        replace_smaller_stride(myPoint, index+1, stride, *global_point);
    }
    // replace the data if bigger
    for(i=k*threadNum+threadId; i<length; i+=threadNum)
    {
        replace_smaller_stride(myPoint, k, stride, input[i]);
    }
    __syncthreads();
#if 1
    for(i=localThreadNum>>1; i>0; i>>=1) {
        if(localThreadId < i)
        {
            mergeTwoK_stride(myPoint, myPoint+i, k, stride);
        }
        __syncthreads();
    }
#endif

    if(threadIdx.x == 0)
    {
        current = myPoint;
        // produce k data in decent order
        DATATYPE *outputPoint = input + blockIdx.x*localThreadNum*k;
        for(i=0; i<k; i++,current+=stride)
        {
            outputPoint[i] = *current;
        }
    }
}
__global__ void top_k_gpu_kernel4_2(DATATYPE* input, int num, int skip_stride, int k, DATATYPE* output)
{
    extern __shared__ DATATYPE shared_buffer[];
    DATATYPE *myPoint = shared_buffer + threadIdx.x;
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    int threadNum = gridDim.x * blockDim.x;
    int localThreadId = threadIdx.x;
    int localThreadNum = blockDim.x;
    int stride = localThreadNum; 
    int i;

    for(i=0; i<k; i++)
        myPoint[i*stride] = input[threadId*skip_stride + i];
    for(i=threadNum+threadId; i<num; i+=threadNum)
        mergeTwoK_stride_lin(myPoint, input+i*skip_stride, k, stride);
    __syncthreads();

    // reduction
    for(i=localThreadNum>>1; i>0; i>>=1) {
        if(localThreadId < i)
        {
            mergeTwoK_stride(myPoint, myPoint+i, k, stride);
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
    {
        // produce k data in decent order
        DATATYPE *outputPoint = output + blockIdx.x*localThreadNum*skip_stride;
        DATATYPE *current = myPoint;
        for(i=0; i<k; i++,current+=stride)
        {
            outputPoint[i] = *current;
        }
    }
}




void top_k_gpu(DATATYPE* input, int length, int k, DATATYPE* output)
{
#if 0
    // k < 12
    int blocks = 1;
    int threads = (GPU_THREADS < length/(4*k)*2) ? GPU_THREADS : (length/(4*k)*2);
    top_k_gpu_kernel1<<<blocks, threads>>>(input, length, k, output);
    cudaError_t err = cudaGetLastError();
    HANDLE_CUDA_ERROR(err);
#endif

#if 0
    // k < 12
    int blocks = 1;
    int threads = (GPU_THREADS < length/(4*k)*2) ? GPU_THREADS : (length/(4*k)*2);
    int shared_mem_usage = sizeof(DATATYPE)*k*threads;
    top_k_gpu_kernel2<<<blocks, threads, shared_mem_usage>>>(input, length, k, output);
    cudaError_t err = cudaGetLastError();
    HANDLE_CUDA_ERROR(err);
#endif

#if 1
    // each thread at least 2K
    int blocks_opt, thread_opt;
    if(k < 20)
    {
        blocks_opt = GPU_BLOCKS_THRESHOLD;
        thread_opt = GPU_THREADS;
    }
    else{
        blocks_opt = 16;
        thread_opt = 64;
    }
    int threads = (thread_opt < length/(4*k)*2) ? thread_opt : (length/(4*k)*2);
    int stride = threads * k;
    int blocks = (blocks_opt < length / (threads*2*k)) ? blocks_opt : (length / (threads*2*k));
    int shared_mem_usage = sizeof(DATATYPE)*k*threads;
    //printf("shared mem usage: (%d %d) %d(%d)\n", blocks, threads, shared_mem_usage, GPU_SHARED_MEM_THRESHOLD);
    if(shared_mem_usage < GPU_SHARED_MEM_THRESHOLD)
        top_k_gpu_kernel3_1<<<blocks, threads, shared_mem_usage>>>(input, length, k, output);
    else
        top_k_gpu_kernel3_1_orig<<<blocks, threads>>>(input, length, k, output);
    threads = (thread_opt < blocks / 2) ? thread_opt : (blocks / 2);
    shared_mem_usage = sizeof(DATATYPE)*k*threads;
    //printf("shared mem usage: (%d %d) %d(%d)\n", 1, threads, shared_mem_usage, GPU_SHARED_MEM_THRESHOLD);
    if(shared_mem_usage < GPU_SHARED_MEM_THRESHOLD)
        top_k_gpu_kernel3_2<<<1, threads, shared_mem_usage>>>(input, blocks, stride, k, output);
    else
        top_k_gpu_kernel3_2_orig<<<1, threads>>>(input, blocks, stride, k, output);
    cudaError_t err = cudaGetLastError();
    HANDLE_CUDA_ERROR(err);
#endif
#if 0
    // k < 12
    int threads = (GPU_THREADS < length/(4*k)*2) ? GPU_THREADS : (length/(4*k)*2);
    int stride = threads * k;
    int blocks = (GPU_BLOCKS_THRESHOLD < length / (threads*2*k)) ? GPU_BLOCKS_THRESHOLD : (length / (threads*2*k));
    int shared_mem_usage = sizeof(DATATYPE)*k*threads;
    if(shared_mem_usage < GPU_SHARED_MEM_THRESHOLD)
        top_k_gpu_kernel4_1<<<blocks, threads, shared_mem_usage>>>(input, length, k, output);
    else
        printf("%d %d %d\n", blocks, threads, shared_mem_usage);
    threads = (GPU_THREADS < blocks / 2) ? GPU_THREADS : (blocks / 2);
    shared_mem_usage = sizeof(DATATYPE)*k*threads;
    if(shared_mem_usage < GPU_SHARED_MEM_THRESHOLD)
        top_k_gpu_kernel4_2<<<1, threads, shared_mem_usage>>>(input, blocks, stride, k, output); 
    else
        printf("%d %d %d\n", blocks, threads, shared_mem_usage);
    cudaError_t err = cudaGetLastError();
    HANDLE_CUDA_ERROR(err);
#endif
}
#endif
