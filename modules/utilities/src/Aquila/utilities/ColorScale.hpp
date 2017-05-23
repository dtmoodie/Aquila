#pragma once
#include "Aquila/core/detail/Export.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace aq
{
    struct AQUILA_EXPORTS ColorScale
    {
	    __host__ __device__ ColorScale(float start_ = 0, float slope_ = 1, bool symmetric_ = false);
	    // Defines where this color starts to take effect, between zero and 1
        float start;
	    // Defines the slope of increase / decrease for this color between 
        float slope;
	    // Defines if the slope decreases after it peaks
	    bool	symmetric;
	    // Defines if this color starts high then goes low
	    bool	inverted;
	    bool flipped;
        void Rescale(float alpha, float beta);
	    __host__ __device__ float operator()(float location);
	    __host__ __device__ float GetValue(float location_);
        template<class A> void serialize(A& ar)
        {
            ar(start);
            ar(slope);
            ar(symmetric);
            ar(inverted);
            ar(flipped);
        }
    };
}
