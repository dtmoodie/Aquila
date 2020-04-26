aquila_declare_module(NAME utilities)

if(BUILD_TESTS)
    set(CUDA_NVCC_FLAGS "-std=c++11")
    file(GLOB_RECURSE test_src "tests/*.cpp" "tests/*.cu")
    cuda_add_executable(test_aq_utilities ${test_src})
    target_link_libraries(test_aq_utilities PRIVATE aquila_utilities ${OpenCV_LIBRARIES})
endif()

