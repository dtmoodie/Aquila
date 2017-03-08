#include "MetaObject/Logging/Profiling.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Thread/ThreadRegistry.hpp"
#include <string>
#include <sstream>
#include "nvToolsExt.h"
#include "nvToolsExtCuda.h"
#ifdef HAVE_CUDA
#include "cuda.h"
  #ifdef HAVE_OPENCV
    #include <opencv2/core/cuda_stream_accessor.hpp>
  #endif
#endif
#if WIN32
  #include "Windows.h"
#else
  #include "dlfcn.h"
#endif
#include "Remotery.h"

using namespace mo;

typedef int(*push_f)(const char*);
typedef int(*pop_f)();
typedef void(*nvtx_name_thread_f)(uint32_t, const char*);
typedef void(*nvtx_name_stream_f)(CUstream , const char*);

typedef void(*rmt_push_cpu_f)(const char*, unsigned int*);
typedef void(*rmt_pop_cpu_f)();
typedef void(*rmt_push_cuda_f)(const char*, unsigned int*, void*);
typedef void(*rmt_pop_cuda_f)(void*);
typedef void(*rmt_set_thread_name_f)(const char*);

#ifndef PROFILING_NONE
push_f nvtx_push = nullptr;
pop_f nvtx_pop = nullptr;
nvtx_name_thread_f nvtx_name_thread = nullptr;
nvtx_name_stream_f nvtx_name_stream = nullptr;

Remotery* rmt = nullptr;

rmt_push_cpu_f rmt_push_cpu = nullptr;
rmt_pop_cpu_f rmt_pop_cpu = nullptr;
rmt_push_cuda_f rmt_push_gpu = nullptr;
rmt_pop_cuda_f rmt_pop_gpu = nullptr;
rmt_set_thread_name_f rmt_set_thread = nullptr;
#endif
void mo::SetThreadName(const char* name)
{
    if(rmt_set_thread)
    {
        rmt_set_thread(name);
    }
    if(nvtx_name_thread)
    {
        nvtx_name_thread(mo::GetThisThread(), name);
    }
}



#ifdef RMT_BUILTIN
void InitRemotery()
{
    if(rmt)
        return;
    rmt_CreateGlobalInstance(&rmt);
    CUcontext ctx;
    cuCtxGetCurrent(&ctx);
    rmtCUDABind bind;
    bind.context = ctx;
    bind.CtxSetCurrent = (void*)&cuCtxSetCurrent;
    bind.CtxGetCurrent = (void*)&cuCtxGetCurrent;
    bind.EventCreate = (void*)&cuEventCreate;
    bind.EventDestroy = (void*)&cuEventDestroy;
    bind.EventRecord = (void*)&cuEventRecord;
    bind.EventQuery = (void*)&cuEventQuery;
    bind.EventElapsedTime = (void*)&cuEventElapsedTime;
    rmt_BindCUDA(&bind);
    rmt_push_cpu = &_rmt_BeginCPUSample;
    rmt_pop_cpu = &_rmt_EndCPUSample;
    rmt_push_gpu = &_rmt_BeginCUDASample;
    rmt_pop_gpu = &_rmt_EndCUDASample;
    rmt_set_thread = &_rmt_SetCurrentThreadName;
}

#else
void InitRemotery()
{
    if(rmt_push_cpu && rmt_pop_cpu)
        return;
#ifdef _DEBUG
    HMODULE handle = LoadLibrary("remoteryd.dll");
#else
    HMODULE handle = LoadLibrary("remotery.dll");
#endif
    if (handle)
    {
        LOG(info) << "Loaded remotery library for profiling";
        typedef void(*rmt_init)(Remotery**);
        rmt_init init = (rmt_init)GetProcAddress(handle, "_rmt_CreateGlobalInstance");
        if (init)
        {
            init(&rmt);
#ifdef HAVE_CUDA
            typedef void(*rmt_cuda_init)(const rmtCUDABind*);
            rmt_cuda_init cuda_init = (rmt_cuda_init)(GetProcAddress(handle, "_rmt_BindCUDA"));
            if (cuda_init)
            {
                CUcontext ctx;
                cuCtxGetCurrent(&ctx);
                rmtCUDABind bind;
                bind.context = ctx;
                bind.CtxSetCurrent = (void*)&cuCtxSetCurrent;
                bind.CtxGetCurrent = (void*)&cuCtxGetCurrent;
                bind.EventCreate = (void*)&cuEventCreate;
                bind.EventDestroy = (void*)&cuEventDestroy;
                bind.EventRecord = (void*)&cuEventRecord;
                bind.EventQuery = (void*)&cuEventQuery;
                bind.EventElapsedTime = (void*)&cuEventElapsedTime;
                cuda_init(&bind);
            }
            rmt_push_cpu = (rmt_push_cpu_f)GetProcAddress(handle, "_rmt_BeginCPUSample");
            rmt_pop_cpu = (rmt_pop_cpu_f)GetProcAddress(handle, "_rmt_EndCPUSample");
            rmt_push_gpu = (rmt_push_cuda_f)GetProcAddress(handle, "_rmt_BeginCUDASample");
            rmt_pop_gpu = (rmt_pop_cuda_f)GetProcAddress(handle, "_rmt_EndCUDASample");
            rmt_set_thread = (rmt_set_thread_name_f)GetProcAddress(handle, "_rmt_SetCurrentThreadName");
#endif
        }
    }
    else
    {
        LOG(info) << "No remotery library found";
    }
}
#endif

void InitNvtx()
{
    if (nvtx_push && nvtx_pop)
        return;
#ifdef _MSC_VER
    HMODULE nvtx_handle = LoadLibrary("nvToolsExt64_1.dll");
    if (nvtx_handle)
    {
        LOG(info) << "Loaded nvtx module";
        nvtx_push = (push_f)GetProcAddress(nvtx_handle, "nvtxRangePushA");
        nvtx_pop = (pop_f)GetProcAddress(nvtx_handle, "nvtxRangePop");
    }
    else
    {
        LOG(info) << "No nvtx library loaded";
    }
#else
    void* nvtx_handle = dlopen("libnvToolsExt.so", RTLD_NOW);
    if (nvtx_handle)
    {
        LOG(info) << "Loaded nvtx module";
        nvtx_push = (push_f)dlsym(nvtx_handle, "nvtxRangePushA");
        nvtx_pop = (pop_f)dlsym(nvtx_handle, "nvtxRangePop");
        nvtx_name_thread = (nvtx_name_thread_f)dlsym(nvtx_handle, "nvtxNameOsThreadA");
        nvtx_name_stream = (nvtx_name_stream_f)dlsym(nvtx_handle, "nvtxNameCuStreamA");
    }
    else
    {
        LOG(info) << "No nvtx library loaded";
    }
#endif
}


void mo::InitProfiling()
{
#ifndef PROFILING_NONE
    InitNvtx();
    InitRemotery();
#endif
}
void mo::PushCpu(const char* name, unsigned int* rmt_hash)
{
    if (nvtx_push)
        (*nvtx_push)(name);
    if (rmt && rmt_push_cpu)
    {
        rmt_push_cpu(name, rmt_hash);
    }
}

void mo::PopCpu()
{
    if (nvtx_pop)
    {
        (*nvtx_pop)();
    }
    if (rmt && rmt_pop_cpu)
    {
        rmt_pop_cpu();
    }
}
void mo::SetStreamName(const char* name, cv::cuda::Stream& stream)
{
    if(nvtx_name_stream)
    {
        nvtx_name_stream(cv::cuda::StreamAccessor::getStream(stream), name);
    }
}

scoped_profile::scoped_profile(std::string name, unsigned int* obj_hash, unsigned int* cuda_hash, cv::cuda::Stream* stream):
    scoped_profile(name.c_str(), obj_hash, cuda_hash, stream)
{

}

scoped_profile::scoped_profile(const char* name, unsigned int* rmt_hash, unsigned int* rmt_cuda, cv::cuda::Stream* stream)
{
#ifndef PROFILING_NONE
    if(nvtx_push)
        (*nvtx_push)(name);
	if (rmt && rmt_push_cpu)
	{
        rmt_push_cpu(name, rmt_hash);
	}
#endif
}

scoped_profile::scoped_profile(const char* name, const char* func, unsigned int* rmt_hash, unsigned int* rmt_cuda, cv::cuda::Stream* stream)
{
#ifndef PROFILING_NONE
	std::stringstream ss;
	ss << name;
	ss << "[";
	ss << func;
	ss << "]";
    const char* str = ss.str().c_str();
    if(nvtx_push)
    {
        (*nvtx_push)(str);
    }
	if (rmt && rmt_push_cpu)
	{
        rmt_push_cpu(str, rmt_hash);
        if(stream && rmt_push_gpu)
        {
            rmt_push_gpu(str, rmt_cuda, cv::cuda::StreamAccessor::getStream(*stream));
            this->stream = stream;
        }
	}
#endif
}

scoped_profile::~scoped_profile()
{
#ifndef PROFILING_NONE
    if(nvtx_pop)
    {
        (*nvtx_pop)();
    }
	if (rmt && rmt_pop_cpu)
	{
        rmt_pop_cpu();
	}
    if(stream && rmt_pop_gpu)
    {
        rmt_pop_gpu(cv::cuda::StreamAccessor::getStream(*stream));
    }
#endif
}
