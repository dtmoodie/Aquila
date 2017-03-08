file(GLOB src "*.cpp")
file(GLOB hdr "*.hpp")

add_executable(console_reflection ${hdr} ${src})
target_link_libraries(console_reflection MetaObject)
set_target_properties(console_reflection PROPERTIES FOLDER examples)
if(MSVC)
      CONFIGURE_FILE("../../tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/examples/console_reflection/console_reflection.vcxproj.user @ONLY)
endif()
