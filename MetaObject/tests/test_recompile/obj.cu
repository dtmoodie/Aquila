#ifdef HAVE_CUDA

#include "obj.h"
#include <thrust/device_vector.h>
#include <thrust/fill.h>

using namespace mo;





void test_cuda_object::run_kernel()
{
    thrust::device_vector<int> D(10,1);
    thrust::fill(D.begin(), D.end(), 9);
}

#endif