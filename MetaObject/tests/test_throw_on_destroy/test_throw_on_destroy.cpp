
#define BOOST_TEST_MAIN


#include "MetaObject/Logging/Log.hpp"


#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif
#include <iostream>

using namespace mo;

bool throwing_function()
{
    THROW(debug) << "throwing from function";
    return false;
}

BOOST_AUTO_TEST_CASE(signals)
{
    BOOST_REQUIRE_THROW(THROW(debug) << "test", mo::ExceptionWithCallStack<std::string>);
    BOOST_REQUIRE_THROW(ASSERT_EQ(throwing_function(), true), mo::ExceptionWithCallStack<std::string>);
}
