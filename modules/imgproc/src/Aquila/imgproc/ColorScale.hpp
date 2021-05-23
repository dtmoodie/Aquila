#pragma once
#include "Aquila/detail/export.hpp"
#include <cuda.h>
#include <cuda_runtime.h>

namespace aq
{
    struct AQUILA_EXPORTS ColorScale
    {
        inline ColorScale(float start_ = 0, float slope_ = 1, bool symmetric_ = false)
        {
            start = start_;
            slope = slope_;
            symmetric = symmetric_;
            flipped = false;
            inverted = false;
        }
        
        // Defines where this color starts to take effect, between zero and 1
        float start;
        // Defines the slope of increase / decrease for this color between
        float slope;
        // Defines if the slope decreases after it peaks
        bool symmetric;
        // Defines if this color starts high then goes low
        bool inverted;
        bool flipped;

        void rescale(float alpha, float beta)
        {
            start = start * alpha - beta;
            slope *= alpha;
        }
        
        inline float operator()(float location)
        {
            return getValue(location);
        }

        inline float getValue(float location_)
        {
            float value = 0;
            if (location_ > start)
            {
                value = (location_ - start)*slope;
            }
            else
            {
                value = 0;
            }
            if (value > 1.0f)
            {
                if (symmetric) value = 2.0f - value;
                else value = 1.0f;
            }
            if (value < 0) value = 0;
            if (inverted) value = 1.0f - value;
            return value;
        }
        
        template <class A>
        void serialize(A& ar)
        {
            ar(start);
            ar(slope);
            ar(symmetric);
            ar(inverted);
            ar(flipped);
        }
    };
}
