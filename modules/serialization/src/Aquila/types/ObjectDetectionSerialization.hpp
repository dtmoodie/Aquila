#pragma once
#include "Aquila/types/ObjectDetection.hpp"
#include <Aquila/serialization/cereal/eigen.hpp>

#include <cereal/cereal.hpp>
#include <cereal/types/boost/optional.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

namespace aq
{
    template <class AR>
    void serialize(AR& ar, const Category* cat)
    {
        if (cat)
        {
            ar(cereal::make_nvp("name", cat->name));
        }
    }
}
