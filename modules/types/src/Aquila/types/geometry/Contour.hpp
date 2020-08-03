#ifndef AQUILA_CONTOUR_HPP
#define AQUILA_CONTOUR_HPP
#include <opencv2/core/types.hpp>

#include <ct/reflect.hpp>
#include <ct/types/opencv.hpp>

#include <vector>
namespace aq
{
    using Contour = ct::TArrayView<cv::Point>;
    /*{
        template <class... ARGS>
        Contour(ARGS&&... args)
            : points(std::forward<ARGS>(args)...)
        {
        }

         points;
    };*/
} // namespace aq

/*namespace ct
{
    REFLECT_DERIVED(aq::Contour)
        PUBLIC_ACCESS(points)
    REFLECT_END;
} // namespace ct
*/
#endif // AQUILA_CONTOUR_HPP