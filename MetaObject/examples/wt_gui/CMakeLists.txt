add_executable(wt_gui main.cpp)
target_link_libraries(wt_gui 
  MetaObject
  debug;${Wt_HTTP_DEBUG_LIBRARY}
  debug;${Wt_DEBUG_LIBRARY}
  optimized;${Wt_LIBRARY}
  optimized;${Wt_HTTP_LIBRARY}
)
