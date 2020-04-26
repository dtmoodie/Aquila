#pragma once
#include <Eigen/Core>
#include <ct/reflect.hpp>

namespace aq
{
template <class T>
struct Circle
{

    Circle(T x = 0.0, T y = 0.0, T rad = 0) : origin(x, y), radius(rad) {}

    Circle(Eigen::Matrix<T, 2, 1> og, T rad = 0.0) : origin(og), radius(rad) {}

    bool operator==(const aq::Circle<float>& other) { return origin == other.origin && radius == other.radius; }
    Eigen::Matrix<T, 2, 1> origin;
    T radius;
};
}

namespace ct
{
REFLECT_TEMPLATED_START(aq::Circle)
    PUBLIC_ACCESS(radius)
// PUBLIC_ACCESS(origin)
REFLECT_END;

}
