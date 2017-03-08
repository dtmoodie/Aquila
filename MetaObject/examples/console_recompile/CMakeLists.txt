file(GLOB src "*.cpp")
file(GLOB hdr "*.hpp")

add_executable(console_recompile ${hdr} ${src})
target_link_libraries(console_recompile MetaObject)
set_target_properties(console_recompile PROPERTIES FOLDER examples)
if(MSVC)
      CONFIGURE_FILE("../../tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/examples/console_recompile/console_recompile.vcxproj.user @ONLY)
endif()
