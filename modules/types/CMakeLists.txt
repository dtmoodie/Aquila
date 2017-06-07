find_package(Eigen3 REQUIRED)
aquila_declare_module(NAME types DEPENDS opencv_core aquila_core CXX_STANDARD 14)

if(EIGEN_INCLUDE_DIRS)
    target_include_directories(aquila_types
        PUBLIC $<BUILD_INTERFACE:${EIGEN_INCLUDE_DIRS}>
    )
endif()

if(EIGEN3_INCLUDE_DIR)
    target_include_directories(aquila_types
        PUBLIC $<BUILD_INTERFACE:${EIGEN3_INCLUDE_DIR}>
    )
endif()
