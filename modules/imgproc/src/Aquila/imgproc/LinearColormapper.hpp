#pragma once
#include "ColorScale.hpp"
#include "IColorMapper.hpp"
namespace aq
{
    // Uses the ColorScale object to create linear mappings for red, green, and blue separately
    class AQUILA_EXPORTS LinearColorMapper : public IColorMapper
    {
      public:
        LinearColorMapper();
        LinearColorMapper(const ColorScale& red, const ColorScale& green, const ColorScale& blue);
        virtual void Apply(cv::InputArray input,
                           cv::OutputArray output,
                           cv::InputArray mask = cv::noArray(),
                           cv::cuda::Stream& stream = cv::cuda::Stream::Null());
        virtual cv::Mat_<float> getMat(float min, float max, int resolution);
        virtual void Rescale(float, float);
        template <typename A>
        void serialize(A& ar)
        {
            ar(_red);
            ar(_green);
            ar(_blue);
        }

      private:
        ColorScale _red;
        ColorScale _green;
        ColorScale _blue;
    };
} // namespace aq