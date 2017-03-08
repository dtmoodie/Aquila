#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <string>
namespace cv
{
	namespace cuda
	{
		class Stream;
	}
}

namespace mo
{
    MO_EXPORTS void SetThreadName(const char* name);
	class IMetaObject;
    MO_EXPORTS void InitProfiling();
    MO_EXPORTS void PushCpu(const char* name, unsigned int* rmt_hash = nullptr);
    MO_EXPORTS void PopCpu();
    MO_EXPORTS void SetStreamName(const char* name, cv::cuda::Stream& stream);
    struct MO_EXPORTS scoped_profile
    {
        scoped_profile(std::string name, unsigned int* obj_hash = nullptr, unsigned int* cuda_hash = nullptr, cv::cuda::Stream* stream = nullptr);
        scoped_profile(const char* name, unsigned int* obj_hash = nullptr, unsigned int* cuda_hash = nullptr, cv::cuda::Stream* stream = nullptr);
        scoped_profile(const char* name, const char* func, unsigned int* obj_hash = nullptr, unsigned int* cuda_hash = nullptr, cv::cuda::Stream* stream = nullptr);
        ~scoped_profile();
    private:
        cv::cuda::Stream* stream = nullptr;
    };
}


#define PROFILE_OBJ(name) \
mo::scoped_profile profile_object(name, __FUNCTION__, &_rmt_hash, &_rmt_cuda_hash, _ctx->stream)

#define PROFILE_RANGE(name) \
mo::scoped_profile profile_scope_##name(#name)

#define PROFILE_FUNCTION \
mo::scoped_profile profile_function(__FUNCTION__);
