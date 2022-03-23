#ifndef AQ_UTILITIES_CONTAINER_HPP
#define AQ_UTILITIES_CONTAINER_HPP
#include <algorithm>

namespace aq
{
    template <class CONTAINER, class ELEMENT>
    bool contains(const CONTAINER& container, const ELEMENT& element)
    {
        return std::find(container.begin(), container.end(), element) != container.end();
    }

    template <class CONTAINER, class ELEMENT>
    auto find(const CONTAINER& container, const ELEMENT& element)
    {
        return std::find(container.begin(), container.end(), element);
    }

    template <class CONTAINER, class ELEMENT>
    auto find(CONTAINER& container, const ELEMENT& element)
    {
        return std::find(container.begin(), container.end(), element);
    }
} // namespace aq

#endif // AQ_UTILITIES_CONTAINER_HPP
