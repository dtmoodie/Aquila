set(METAOBJECT_MODULES_LIST "" CACHE STRING BOOL FORCE)
function(metaobject_declare_module)
    set(oneValueArgs NAME)
    set(multiValueArgs SRC DEPENDS FLAGS CUDA_SRC INCLUDES)
    cmake_parse_arguments(metaobject_declare_module "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    GroupSources("${CMAKE_CURRENT_LIST_DIR}/src/MetaObject/" "/" " ")
    if(${metaobject_declare_module_CUDA_SRC})
        cuda_add_library(metaobject_${metaobject_declare_module_NAME} SHARED ${${metaobject_declare_module_CUDA_SRC}})
    else()
        if(${metaobject_declare_module_SRC})
            add_library(metaobject_${metaobject_declare_module_NAME} SHARED ${${metaobject_declare_module_SRC}})
        else()
            file(GLOB_RECURSE src "src/*.cpp" "src/*.h" "src/*.hpp")
            add_library(metaobject_${metaobject_declare_module_NAME} SHARED ${src})
        endif()
    endif()
    
    set_target_properties(metaobject_${metaobject_declare_module_NAME}
        PROPERTIES
        LINKER_LANGUAGE CXX
    )
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" )
        # This is a workaround for cotire not correctly pulling the cxx standard that is inherited from ct
        # https://github.com/sakra/cotire/issues/138
        if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
            set_target_properties(metaobject_${metaobject_declare_module_NAME}
                PROPERTIES
                CXX_STANDARD 14
            )
        else()
            set_target_properties(metaobject_${metaobject_declare_module_NAME}
                PROPERTIES
                CXX_STANDARD 11
            )
        endif()
    endif()
    
    set(metaobject_modules "${metaobject_modules};metaobject_${metaobject_declare_module_NAME}" CACHE INTERNAL "" FORCE)

    target_compile_definitions(metaobject_${metaobject_declare_module_NAME} PRIVATE -DMetaObject_EXPORTS)

    if(metaobject_declare_module_DEPENDS)
        target_link_libraries(metaobject_${metaobject_declare_module_NAME}
            PUBLIC ${metaobject_declare_module_DEPENDS}
        )
    endif()
    if(metaobject_declare_module_INCLUDES)
        target_include_directories(metaobject_python
            PUBLIC
                ${metaobject_declare_module_INCLUDES}
        )
    endif()

    if(UNIX)
        target_compile_options(metaobject_${metaobject_declare_module_NAME} PUBLIC "-fPIC;-Wl,--no-undefined")
    endif()
    set(metaobject_${metaobject_declare_module_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src" CACHE INTERNAL "" FORCE)
    target_include_directories(metaobject_${metaobject_declare_module_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
            $<INSTALL_INTERFACE:include>
    )
    set(metaobject_module_includes "${metaobject_module_includes};${CMAKE_CURRENT_LIST_DIR}/src" CACHE INTERNAL "" FORCE)

    set_target_properties(metaobject_${metaobject_declare_module_NAME} PROPERTIES FOLDER Modules)
    if(metaobject_declare_module_FLAGS)
        target_compile_options(metaobject_${metaobject_declare_module_NAME} PUBLIC ${metaobject_declare_module_FLAGS})
    endif()

    if(EXISTS "${CMAKE_CURRENT_LIST_DIR}/pch.hpp")
        configure_file(${CMAKE_CURRENT_LIST_DIR}/pch.hpp
            ${CMAKE_BINARY_DIR}/include/MetaObject/${metaobject_declare_module_NAME}/${metaobject_declare_module_NAME}_pch.hpp @ONLY)
    endif()

    export(TARGETS metaobject_${metaobject_declare_module_NAME}
        FILE "${PROJECT_BINARY_DIR}/MetaObjectTargets-${metaobject_declare_module_NAME}.cmake"
    )
    install(TARGETS metaobject_${metaobject_declare_module_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(TARGETS metaobject_${metaobject_declare_module_NAME}
        DESTINATION lib
        EXPORT metaobject_${metaobject_declare_module_NAME}Targets
    )
    install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
    )

    install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp"
    )
    install(EXPORT metaobject_${metaobject_declare_module_NAME}Targets DESTINATION "${CMAKE_INSTALL_PREFIX}/share/MetaObject" COMPONENT dev)

    if(RCC_VERBOSE_CONFIG)
        set(test_lib "")
        set(test_inc "")
        set(test_lib_debug "")
        set(test_lib_release "")
        set(test_deps "")
        set(test_flags "")
        _target_helper(test_lib test_inc test_lib_debug test_lib_release test_deps test_flags metaobject_${metaobject_declare_module_NAME} " ")
        message("---------------")
        list(REMOVE_DUPLICATES test_lib)
        foreach(lib ${test_lib})
            message("  ${lib}")
        endforeach()
        list(REMOVE_DUPLICATES test_inc)
        foreach(inc ${test_inc})
            message("  ${inc}")
        endforeach()
    endif(RCC_VERBOSE_CONFIG)
    set(METAOBJECT_MODULES_LIST "${metaobject_declare_module_NAME};${METAOBJECT_MODULES_LIST}" CACHE STRING BOOL FORCE)    
endfunction()
