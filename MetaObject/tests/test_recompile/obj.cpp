#include "obj.h"
void test_meta_object_slots::test_void()
{
    ++call_count;
}

void test_meta_object_slots::test_int(int value)
{
    call_count += value;
}


MO_REGISTER_OBJECT(test_meta_object_signals);
MO_REGISTER_OBJECT(test_meta_object_slots);
MO_REGISTER_OBJECT(test_meta_object_parameters);
MO_REGISTER_OBJECT(test_meta_object_output);
MO_REGISTER_OBJECT(test_meta_object_input);
#ifdef HAVE_CUDA
MO_REGISTER_OBJECT(test_cuda_object);
#endif