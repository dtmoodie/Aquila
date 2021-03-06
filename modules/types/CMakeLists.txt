find_package(Eigen3 REQUIRED)

if(NOT TARGET Eigen3)
    add_library(Eigen3 IMPORTED INTERFACE)
    if(EIGEN3_INCLUDE_DIR)
        set_target_properties(Eigen3 PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    endif()

    if(EIGEN_INCLUDE_DIRS)
        set_target_properties(Eigen3 PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${EIGEN_INCLUDE_DIRS}"
        )
    endif()
endif()

aquila_declare_module(NAME types DEPENDS
    opencv_core
    opencv_imgproc
    metaobject_types
    metaobject_params
    aquila_core
    Eigen3
    ce
    CXX_STANDARD 11
)

target_include_directories(aquila_types
    PUBLIC
        $<BUILD_INTERFACE:${aquila_core_INCLUDE_DIRS}>
        $<BUILD_INTERFACE:${aquila_rcc_INCLUDE_DIRS}>
)


if(BUILD_TESTS)
    set(CUDA_NVCC_FLAGS "-std=c++11")
    file(GLOB_RECURSE test_src "tests/*.cpp" "tests/*.cu")

    cuda_add_executable(test_aq_types ${test_src})
    target_link_libraries(test_aq_types PRIVATE aquila_types metaobject_cuda gtest gtest_main)
    set_target_properties(test_aq_types PROPERTIES FOLDER Tests/Aquila)
    add_test(test_aq_types test_aq_types)

    if(OpenCV_FOUND)
        find_package(OpenCV REQUIRED COMPONENTS cudaarithm imgcodecs)
        target_compile_definitions(test_aq_types PUBLIC HAVE_OPENCV)
        target_link_libraries(test_aq_types PRIVATE ${OpenCV_LIBRARIES})
    endif()
    file(COPY "data/data.png" DESTINATION "${CMAKE_BINARY_DIR}")
endif()

