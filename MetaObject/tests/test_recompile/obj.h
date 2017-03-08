#pragma once

#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Parameters/ParameterMacros.hpp"

#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"

#include "RuntimeInclude.h"
#include "RuntimeSourceDependency.h"
RUNTIME_MODIFIABLE_INCLUDE 

#ifdef HAVE_CUDA
RUNTIME_COMPILER_SOURCEDEPENDENCY_EXT(".cu")
#endif

using namespace mo;
struct test_meta_object_signals: public IMetaObject
{
    ~test_meta_object_signals()
    {
        std::cout << "Deleting object\n";
    }
    MO_BEGIN(test_meta_object_signals)
    MO_SIGNAL(void, test_void)
    MO_SIGNAL(void, test_int, int)
    MO_END
};

struct test_meta_object_slots: public IMetaObject
{
    MO_BEGIN(test_meta_object_slots)
    MO_SLOT(void, test_void)
    MO_SLOT(void, test_int, int)
    PROPERTY(int, call_count, 0)
    MO_END
};

struct test_meta_object_parameters: public IMetaObject
{
    MO_BEGIN(test_meta_object_parameters)
    PARAM(int, test, 5)
    MO_END
};

struct test_meta_object_output: public IMetaObject
{
	MO_BEGIN(test_meta_object_output)
        OUTPUT(int, test_output, 0)
    MO_END
};

struct test_meta_object_input: public IMetaObject
{
	MO_BEGIN(test_meta_object_input)
		INPUT(int, test_input, nullptr)
    MO_END
};

#ifdef HAVE_CUDA
struct test_cuda_object: public IMetaObject
{
    MO_BEGIN(test_cuda_object)
    PARAM(int, test, 0)
    MO_END
    void run_kernel();
};
#endif
