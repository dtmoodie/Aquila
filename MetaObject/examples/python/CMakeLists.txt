
add_library(MetaPython SHARED "MetaPython.cpp")

target_link_libraries(MetaPython 
    MetaObject 
    ${Boost_PYTHON3_LIBRARY_DEBUG} 
    ${Boost_PYTHON3_LIBRARY_RELEASE}
    ${PYTHON_LIBRARY})
	
if(PYTHON_DEBUG_LIBRARY)
target_link_libraries(MetaPython 
    ${PYTHON_DEBUG_LIBRARY})
endif()
set_target_properties(MetaPython PROPERTIES FOLDER examples)
set_target_properties(MetaPython PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})