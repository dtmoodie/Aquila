IF(UNIX)
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
  set(CUDA_NVCC_FLAGS "-std=c++11;--expt-relaxed-constexpr;${CUDA_NVCC_FLAGS}")
ENDIF()
aquila_declare_module(NAME utilities DEPENDS opencv_core aquila_core CXX_STANDARD 11)
set_target_properties(aquila_utilities PROPERTIES CXX_STANDARD_REQUIRED ON)

