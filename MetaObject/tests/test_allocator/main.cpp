#define BOOST_TEST_MAIN
#include "MetaObject/Detail/Allocator.hpp"
#include "MetaObject/Detail/AllocatorImpl.hpp"
#include "MetaObject/Logging/Profiling.hpp"

#include <boost/log/core.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks.hpp>
#include <boost/log/attributes.hpp>
#include <boost/log/common.hpp>
#include <boost/log/exceptions.hpp>
#include <opencv2/cudaarithm.hpp>

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "MetaObjectInheritance"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

using namespace mo;

BOOST_AUTO_TEST_CASE(initialize_thread)
{
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::warning);
    BOOST_REQUIRE(mo::Allocator::GetThreadSpecificAllocator());
    mo::InitLogging();
    mo::InitProfiling();
}

BOOST_AUTO_TEST_CASE(initialize_global)
{
    BOOST_REQUIRE(mo::Allocator::GetThreadSafeAllocator());
}



BOOST_AUTO_TEST_CASE(test_cpu_pooled_allocation)
{
    auto start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    auto end = boost::posix_time::microsec_clock::local_time();
    double non_pinned_time = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    mo::PinnedAllocator pinnedAllocator;
    cv::Mat::setDefaultAllocator(&pinnedAllocator);
    for(int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double non_pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();
    mo::CpuPoolPolicy allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();
    
    

    // Test the thread safe allocators
    mo::mt_CpuPoolPolicy mtPoolAllocator;
    cv::Mat::setDefaultAllocator(&mtPoolAllocator);

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(1, 100 * std::min(1000, 1 + rand()), CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double mt_pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();
    cv::Mat::setDefaultAllocator(nullptr);

    BOOST_REQUIRE_LT(pooled_time, non_pooled_time);
    std::cout << "\n ======================================================================== \n";
    std::cout << " Random Allocation Pattern\n";
    std::cout 
        << " Default Allocator Time: " << non_pinned_time << "\n"
        << " Pinned Allocator Time:  " << non_pooled_time << "\n"
        << " Pooled Time:            " << pooled_time << "\n"
        << " Thead Safe Pooled Time: " << mt_pooled_time;
}

BOOST_AUTO_TEST_CASE(test_cpu_stack_allocation)
{
    auto start = boost::posix_time::microsec_clock::local_time();
    cv::Mat zeroAlloc(2000, 2000, CV_32FC2);
    for (int i = 0; i < 1000; ++i)
    {
        zeroAlloc *= 100;
        zeroAlloc += 10;
    }
    auto end = boost::posix_time::microsec_clock::local_time();
    double zero_alloc_time = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double non_pinned_time = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    mo::PinnedAllocator pinnedAllocator;
    cv::Mat::setDefaultAllocator(&pinnedAllocator);
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double non_pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();
    mo::CpuStackPolicy allocator;
    cv::Mat::setDefaultAllocator(&allocator);

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();

    // Test the thread safe allocators
    mo::mt_CpuStackPolicy mtPoolAllocator;
    cv::Mat::setDefaultAllocator(&mtPoolAllocator);

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32FC2);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double mt_pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();
    cv::Mat::setDefaultAllocator(nullptr);

    BOOST_REQUIRE_LT(pooled_time, non_pooled_time);
    std::cout << "\n ======================================================================== \n";
    std::cout << " Fixed Allocation Pattern\n";
    std::cout 
        << " Default Allocator Time:   " << non_pinned_time << "\n"
        << " Pinned Allocator Time:    " << non_pooled_time << "\n"
        << " Pooled Time:              " << pooled_time << "\n"
        << " Thead Safe Pooled Time:   " << mt_pooled_time << "\n"
        << " Zero Allocation Time:     " << zero_alloc_time;
}
BOOST_AUTO_TEST_CASE(test_cpu_combined_allocation)
{
    cv::Mat::setDefaultAllocator(mo::Allocator::GetThreadSpecificAllocator());
    auto start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2, std::min(1000, 1 + rand()) * 100, CV_32F);
        vec *= 100;
        vec += 10;
    }
    auto end = boost::posix_time::microsec_clock::local_time();
    double random_size = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double set_size = boost::posix_time::time_duration(end - start).total_milliseconds();

    cv::Mat::setDefaultAllocator(mo::Allocator::GetThreadSafeAllocator());
    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2, std::min(1000, 1 + rand()) * 100, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double mt_random_size = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        cv::Mat vec(2000, 2000, CV_32F);
        vec *= 100;
        vec += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double mt_set_size = boost::posix_time::time_duration(end - start).total_milliseconds();


    std::cout << "\n ======================================================================== \n";
    std::cout
        << "------------ Thread specifc ---------------\n"
        << " Random Allocation Pattern: " << random_size << "\n"
        << " Set Allocation Pattern:    " << set_size << "\n"
        << "------------ Thread safe ---------------\n"
        << " Random Allocation Pattern: " << mt_random_size << "\n"
        << " Set Allocation Pattern:    " << mt_set_size << "\n";
}

BOOST_AUTO_TEST_CASE(test_gpu_random_allocation_pattern)
{
    cv::cuda::Stream stream;


    cv::cuda::GpuMat X_(1, 1000, CV_32F);
    cv::cuda::GpuMat Y_(1, 1000, CV_32F);
    auto start = boost::posix_time::microsec_clock::local_time();
    for (int i = 0; i < 1000; ++i)
    {
        int cols = std::min(1000, 1 + rand());
        cv::cuda::GpuMat X = X_.colRange(0, cols);
        cv::cuda::GpuMat Y = Y_.colRange(0, cols);
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }
    auto end = boost::posix_time::microsec_clock::local_time();
    double zero_allocator = boost::posix_time::time_duration(end - start).total_milliseconds();

    // Default allocator
    start = boost::posix_time::microsec_clock::local_time();
    for(int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(1, std::min(1000, 1 + rand()), CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }
    end = boost::posix_time::microsec_clock::local_time();
    double default_allocator = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    ConcreteAllocator<h_PoolAllocator_t, d_TensorPoolAllocator_t> poolAllocator;
    auto defaultAllocator = cv::cuda::GpuMat::defaultAllocator();
    cv::cuda::GpuMat::setDefaultAllocator(&poolAllocator);
    for(int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(1, std::min(1000, 1 + rand()), CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }

    end = boost::posix_time::microsec_clock::local_time();
    double pool_allocator = boost::posix_time::time_duration(end - start).total_milliseconds();

    std::cout << "\n ======================= GPU ============================================ \n";
    std::cout
        << "------------ Thread specifc ---------------\n"
        << " Zero Allocation:   " << zero_allocator << "\n"
        << " Default Allocator: " << default_allocator << "\n"
        << " Pool Allocator:    " << pool_allocator << "\n";

    poolAllocator.Release();
    cv::cuda::GpuMat::setDefaultAllocator(defaultAllocator);
    //Allocator::GetThreadSpecificAllocator()->Release();
}


BOOST_AUTO_TEST_CASE(test_gpu_static_allocation_pattern)
{
    boost::log::core::get()->set_filter(boost::log::trivial::severity >= boost::log::trivial::debug);
    cv::cuda::Stream stream;

    // Manual buffer control
    auto start = boost::posix_time::microsec_clock::local_time();
    {
        cv::cuda::GpuMat X(2000, 2000, CV_32F);
        cv::cuda::GpuMat Y(2000, 2000, CV_32F);
        for (int i = 0; i < 1000; ++i)
        {
            cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
            cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
        }
    }
    
    auto end = boost::posix_time::microsec_clock::local_time();
    double zero_allocation = boost::posix_time::time_duration(end - start).total_milliseconds();
    
    // Default allocator
    start = boost::posix_time::microsec_clock::local_time();
    for(int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(2000, 2000, CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }
    end = boost::posix_time::microsec_clock::local_time();
    double default_allocator = boost::posix_time::time_duration(end - start).total_milliseconds();


    // Custom allocator
    ConcreteAllocator<h_PoolAllocator_t, d_TextureAllocator_t> poolAllocator;
    auto defaultAllocator = cv::cuda::GpuMat::defaultAllocator();
    cv::cuda::GpuMat::setDefaultAllocator(&poolAllocator);
    start = boost::posix_time::microsec_clock::local_time();
    for(int i = 0; i < 1000; ++i)
    {
        cv::cuda::GpuMat X(2000, 2000, CV_32F);
        cv::cuda::GpuMat Y;
        cv::cuda::multiply(X, cv::Scalar(100), Y, 1, -1, stream);
        cv::cuda::subtract(Y, cv::Scalar(100), Y, cv::noArray(), -1, stream);
    }

    end = boost::posix_time::microsec_clock::local_time();
    double pool_allocator = boost::posix_time::time_duration(end - start).total_milliseconds();

    std::cout << "\n ======================= GPU ============================================ \n";
    std::cout
        << "------------ Thread specifc ---------------\n"
        << " Zero Allocation:   " << zero_allocation << "\n"
        << " Default Allocator: " << default_allocator << "\n"
        << " Pool Allocator:    " << pool_allocator << "\n";

    poolAllocator.Release();
    Allocator::GetThreadSpecificAllocator()->Release();
    cv::cuda::GpuMat::setDefaultAllocator(defaultAllocator);
}


BOOST_AUTO_TEST_CASE(stl_allocator_pool)
{
    std::vector<float> zero_allocation;
    zero_allocation.resize(2000);
    auto start = boost::posix_time::microsec_clock::local_time();
    for(int i = 0; i < 10000; ++i)
    {
        int size = std::min(2000, rand() + 1);
        cv::Mat view(1, size,CV_32F, zero_allocation.data());
        view *= 100;
        view += 10;
    }
    auto end = boost::posix_time::microsec_clock::local_time();
    double zero_allocation_time = boost::posix_time::time_duration(end - start).total_milliseconds();


    start = boost::posix_time::microsec_clock::local_time();
    for(int i = 0; i < 10000; ++i)
    {
        std::vector<float> vec;
        int size = std::min(2000, rand() + 1);
        vec.resize(size );
        cv::Mat view(1, size,CV_32F, vec.data());
        view *= 100;
        view += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double default_allocation = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    for(int i = 0; i < 10000; ++i)
    {
        std::vector<float, PinnedStlAllocator<float>> vec;
        int size = std::min(2000, rand() + 1);
        vec.resize(size );
        cv::Mat view(1, size, CV_32F, vec.data());
        view *= 100;
        view += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double pinned_allocation = boost::posix_time::time_duration(end - start).total_milliseconds();

    start = boost::posix_time::microsec_clock::local_time();
    for(int i = 0; i < 10000; ++i)
    {
        std::vector<float, PinnedStlAllocatorPoolThread<float>> vec;
        int size = std::min(2000, rand() + 1);
        vec.resize(size );
        cv::Mat view(1, size,CV_32F, vec.data());
        view *= 100;
        view += 10;
    }
    end = boost::posix_time::microsec_clock::local_time();
    double pooled_allocation = boost::posix_time::time_duration(end - start).total_milliseconds();

    std::cout << "\n ======================= STL ============================================ \n";
    std::cout << "Zero Allocation:    " << zero_allocation_time << "\n"
              << "Default Allocation: " << default_allocation << "\n"
              << "Pinned Allocation:  " << pinned_allocation  << "\n"
              << "Pooled Allocation:  " << pooled_allocation << "\n";
}

BOOST_AUTO_TEST_CASE(async_transfer_rate_random)
{
    boost::posix_time::ptime start, end;
    double zero_allocation_time, default_time, pinned_time, pooled_time, stack_time;
    const int num_iterations = 500;
    std::vector<int> sizes;
    sizes.resize(num_iterations);
    std::cout << "\n ======================= H2D -> D2H ============================================ \n";
    for(int i = 0; i < num_iterations; ++i)
    {
        int size = std::min(1000, rand() + 1);
        sizes[i] = size;
    }
    {
        
        std::vector<cv::Mat> h_data;
        std::vector<cv::cuda::GpuMat> d_data;
        std::vector<cv::Mat> result;
        h_data.resize(num_iterations);
        d_data.resize(num_iterations);
        result.resize(num_iterations);
        // Pre allocate
        for(int i = 0; i < num_iterations; ++i)
        {
            h_data[i] = cv::Mat(10, sizes[i], CV_32F);
            cv::cuda::createContinuous(10, sizes[i], CV_32F, d_data[i]);
            result[i] = cv::Mat(10, sizes[i], CV_32F);
        }
        cv::cuda::Stream stream;
        start = boost::posix_time::microsec_clock::local_time();
        mo::scoped_profile profile("zero allocation");
        for(int i = 0; i < num_iterations; ++i)
        {
            d_data[i].upload(h_data[i], stream);
            cv::cuda::multiply(d_data[i], cv::Scalar(100), d_data[i], 1, -1, stream);
            cv::cuda::add(d_data[i], cv::Scalar(100), d_data[i], cv::noArray(), -1, stream);
            d_data[i].download(result[i], stream);
        }
        end = boost::posix_time::microsec_clock::local_time();
        zero_allocation_time = boost::posix_time::time_duration(end - start).total_milliseconds();
        std::cout << "Zero Allocation:    " << zero_allocation_time << "\n";
    }

    {
        
        cv::cuda::Stream stream;
        start = boost::posix_time::microsec_clock::local_time();
        mo::scoped_profile profile("default allocation");
        for(int i = 0; i < sizes.size(); ++i)
        {
            cv::Mat h_data(10, sizes[i], CV_32F);
            cv::cuda::GpuMat d_data;
            d_data.upload(h_data, stream);
            cv::cuda::multiply(d_data, cv::Scalar(100), d_data, 1, -1, stream);
            cv::cuda::add(d_data, cv::Scalar(100), d_data, cv::noArray(), -1, stream);
            cv::Mat result;
            d_data.download(result, stream);
        }
        end = boost::posix_time::microsec_clock::local_time();
        default_time = boost::posix_time::time_duration(end - start).total_milliseconds();
        std::cout << "Default Allocation: " << default_time << "\n";
    }

    {
        mo::ConcreteAllocator<mo::CpuStackPolicy, mo::StackPolicy<cv::cuda::GpuMat, mo::ContinuousPolicy>> allocator;
        auto defaultAllocator = cv::cuda::GpuMat::defaultAllocator();
        cv::cuda::GpuMat::setDefaultAllocator(&allocator);
        cv::Mat::setDefaultAllocator(&allocator);
        cv::cuda::Stream stream;
        start = boost::posix_time::microsec_clock::local_time();
        mo::scoped_profile profile("pool allocation");
        for(int i = 0; i < sizes.size(); ++i)
        {
            cv::Mat h_data(10, sizes[i], CV_32F);
            cv::cuda::GpuMat d_data;
            cv::cuda::createContinuous(10, sizes[i], CV_32F, d_data);
            d_data.upload(h_data, stream);
            cv::cuda::multiply(d_data, cv::Scalar(100), d_data, 1, -1, stream);
            cv::cuda::add(d_data, cv::Scalar(100), d_data, cv::noArray(), -1, stream);
            cv::Mat result;
            d_data.download(result, stream);
        }
        end = boost::posix_time::microsec_clock::local_time();
        pooled_time = boost::posix_time::time_duration(end - start).total_milliseconds();
        std::cout << "Pooled Allocation:  " << pooled_time << "\n";
    }
    
    
              
              
}




