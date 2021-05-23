#pragma once
#include <Aquila/detail/export.hpp>
#include <ct/reflect.hpp>

namespace aq
{
    struct Category;
    struct AQUILA_EXPORTS Classification
    {
        Classification(const Category* cat = nullptr, double conf = 0.0);
        const Category* cat = nullptr;
        double conf = 0.0;
    };
}
namespace ct
{
    REFLECT_BEGIN(aq::Classification)
        PUBLIC_ACCESS(conf)
        PUBLIC_ACCESS(cat)
    REFLECT_END;
}
