#define BOOST_TEST_MAIN

#include "MetaObject/Logging/Log.hpp"
#include <boost/thread.hpp>

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "MetaObjectInheritance"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

BOOST_AUTO_TEST_CASE(logging_init)
{
//    mo::InitLogging();
}

#if BOOST_VERSION > 105800
BOOST_AUTO_TEST_CASE(single_threaded_logging, *boost::unit_test::timeout(100))
{
    for(int i = 0; i < 1000; ++i)
    {
        LOG(debug) << "Test iteration " << i;
    }
}

BOOST_AUTO_TEST_CASE(two_threaded_logging, *boost::unit_test::timeout(100))
{
    boost::thread second_thread(
    []()
    {
        for(int i = 0; i < 10000; ++i)
        {
            LOG(debug) << "Background thread iteration: " << i;
        }
    });
    for (int i = 0; i < 10000; ++i)
    {
        LOG(debug) << "Main thread iteration " << i;
    }
    second_thread.join();
}

BOOST_AUTO_TEST_CASE(ten_threaded_logging, *boost::unit_test::timeout(100))
{
    std::vector<std::shared_ptr<boost::thread>> threads;
    for(int j = 0; j < 10; ++j)
    {
        threads.emplace_back(new boost::thread([j]()
        {
            for (int i = 0; i < 5000; ++i)
            {
                LOG(debug) << "Logging from thread " << j << ": " << i;
            }
        }));
    }
    for (int i = 0; i < 5000; ++i)
    {
        LOG(debug) << "Main thread iteration " << i;
    }
    for(auto thread : threads)
    {
        thread->join();
    }
}

BOOST_AUTO_TEST_CASE(ten_threaded_logging_every_10, *boost::unit_test::timeout(100))
{
    std::vector<std::shared_ptr<boost::thread>> threads;
    for (int j = 0; j < 10; ++j)
    {
        threads.emplace_back(new boost::thread([j]()
        {
            for (int i = 0; i < 5000; ++i)
            {
                LOG_EVERY_N(debug, 10) << "Logging from thread " << j << ": " << i;
            }
        }));
    }
    for (int i = 0; i < 5000; ++i)
    {
        LOG_EVERY_N(debug, 10) << "Main thread iteration " << i;
    }
    for (auto thread : threads)
    {
        thread->join();
    }
}


BOOST_AUTO_TEST_CASE(ten_threaded_logging_first_10, *boost::unit_test::timeout(100))
{
    std::vector<std::shared_ptr<boost::thread>> threads;
    for (int j = 0; j < 10; ++j)
    {
        threads.emplace_back(new boost::thread([j]()
        {
            for (int i = 0; i < 5000; ++i)
            {
                LOG_FIRST_N(debug, 10) << "Logging from thread " << j << ": " << i;
            }
        }));
    }
    for (int i = 0; i < 5000; ++i)
    {
        LOG_FIRST_N(debug, 10) << "Main thread iteration " << i;
    }
    for (auto thread : threads)
    {
        thread->join();
    }
}
#endif
