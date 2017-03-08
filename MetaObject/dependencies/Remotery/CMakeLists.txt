include_directories(lib)
add_definitions(-DRMT_USE_CUDA)
add_library(Remotery SHARED lib/Remotery.c)
set_target_properties(Remotery PROPERTIES COMPILE_DEFINITIONS RMT_DLL)
