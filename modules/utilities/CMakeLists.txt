aquila_declare_module(NAME utilities)

if(BUILD_TESTS)
    # CUDA 13's CCCL headers (thrust/cub/libcu++) require C++17; the project
    # baseline stays C++11 for host code, .cu TUs get C++17
    set(CUDA_NVCC_FLAGS "-std=c++17")
    file(GLOB_RECURSE test_src "tests/*.cpp" "tests/*.cu")
    include_directories(${OpenCV_INCLUDE_DIRS})
    cuda_add_executable(test_aq_utilities ${test_src})
    target_link_libraries(test_aq_utilities PRIVATE aquila_utilities ${OpenCV_LIBRARIES})
endif()

