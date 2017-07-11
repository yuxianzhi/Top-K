#include "top_k.h"
#include "macro.h"

inline void replace_smaller(DATATYPE* array, int k, DATATYPE data)
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
void top_k_cpu_serial(DATATYPE* input, int length, int k, DATATYPE* output)
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
