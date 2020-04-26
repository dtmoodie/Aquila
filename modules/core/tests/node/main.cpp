#include "gtest/gtest.h"
#include <iostream>

#include "common.hpp"
#include "objects.hpp"

using namespace aq;
using namespace aq::nodes;

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    auto system_table = SystemTable::instance();
    mo::MetaObjectFactory::instance()->registerTranslationUnit();
    return RUN_ALL_TESTS();
}
