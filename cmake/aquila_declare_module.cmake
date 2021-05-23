set(ENABLED_AQUILA_MODULES "" CACHE STRING INTERNAL FORCE)
function(aquila_declare_module)
    set(oneValueArgs NAME CXX_STANDARD INCLUDE)
    set(multiValueArgs SRC DEPENDS FLAGS PRIVATE_DEP_HEADER)
    cmake_parse_arguments(aquila_declare_module "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    set(INTERFACE OFF)
    if(${aquila_declare_module_SRC})
        IF(MO_HAVE_CUDA)
            cuda_add_library(aquila_${aquila_declare_module_NAME} SHARED ${${aquila_declare_module_SRC}})
        else()
            add_library(aquila_${aquila_declare_module_NAME} SHARED ${${aquila_declare_module_SRC}})
        endif()
    else()
        if(MO_HAVE_CUDA)
            file(GLOB_RECURSE src "src/*.cpp" "src/*.cu")
            file(GLOB_RECURSE hdr "src/*.h" "src/*.hpp")
            LIST(LENGTH src num_src)
            IF(${num_src} GREATER 0)
                cuda_add_library(aquila_${aquila_declare_module_NAME} SHARED ${src} ${hdr})
            else()
                add_library(aquila_${aquila_declare_module_NAME} INTERFACE)
                add_custom_target(aquila_${aquila_declare_module_NAME}_headers ${hdr})
                set(INTERFACE ON)
            endif()
        else(MO_HAVE_CUDA)
            file(GLOB_RECURSE src "src/*.cpp")
            file(GLOB_RECURSE hdr "src/*.hpp")
            LIST(LENGTH src num_src)
            IF(${num_src} GREATER 0)
                add_library(aquila_${aquila_declare_module_NAME} SHARED ${src})
            else()
                add_library(aquila_${aquila_declare_module_NAME} INTERFACE)
                add_custom_target(aquila_${aquila_declare_module_NAME}_headers ${hdr})
                set(INTERFACE ON)
            endif()
        endif(MO_HAVE_CUDA)
    endif()
    set(aquila_modules "${aquila_modules};aquila_${aquila_declare_module_NAME}" CACHE INTERNAL "" FORCE)
    set(aquila_module_includes "${aquila_module_includes};${CMAKE_CURRENT_LIST_DIR}/src" CACHE INTERNAL "" FORCE)
    set(aquila_${aquila_declare_module_NAME}_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src" CACHE INTERNAL "" FORCE)

    if(aquila_declare_module_DEPENDS)
        target_link_libraries(aquila_${aquila_declare_module_NAME} PUBLIC ${aquila_declare_module_DEPENDS})
        add_dependencies(aquila_${aquila_declare_module_NAME} ${aquila_declare_module_DEPENDS})
    endif()
    if(aquila_declare_module_PRIVATE_DEP_HEADER)
        foreach(dep ${aquila_declare_module_PRIVATE_DEP_HEADER})
            get_target_property(inc ${dep} INTERFACE_INCLUDE_DIRECTORIES)
            if(inc)
                target_include_directories(aquila_${aquila_declare_module_NAME}
                    PRIVATE ${inc})
            else()
                message("No INTERFACE_INCLUDE_DIRECTORIES for ${dep}")
            endif()
        endforeach(dep)
    endif()
    if(INTERFACE)
        target_include_directories(aquila_${aquila_declare_module_NAME}
            INTERFACE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
                $<INSTALL_INTERFACE:include>
        )
    else()
        target_include_directories(aquila_${aquila_declare_module_NAME}
            PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
                $<INSTALL_INTERFACE:include>
        )
    endif()
    if(NOT INTERFACE)
        set_target_properties(aquila_${aquila_declare_module_NAME}
            PROPERTIES
                CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
        )
        set_target_properties(aquila_${aquila_declare_module_NAME} PROPERTIES FOLDER Aquila)
    endif()
    if(aquila_declare_module_CXX_STANDARD)
        set_target_properties(aquila_${aquila_declare_module_NAME} PROPERTIES CXX_STANDARD ${aquila_declare_module_CXX_STANDARD})
    endif()

    if(aquila_declare_module_FLAGS)
        target_compile_options(aquila_${aquila_declare_module_NAME} PUBLIC ${aquila_declare_module_FLAGS})
    endif()

    export(TARGETS aquila_${aquila_declare_module_NAME}
        FILE "${PROJECT_BINARY_DIR}/AquilaTargets-${aquila_declare_module_NAME}.cmake"
    )

    if(NOT INTERFACE)
        install(TARGETS aquila_${aquila_declare_module_NAME}
            RUNTIME DESTINATION bin
            LIBRARY DESTINATION lib
            ARCHIVE DESTINATION lib
        )
    endif()

    install(TARGETS aquila_${aquila_declare_module_NAME}
        DESTINATION lib
        EXPORT aquila_${aquila_declare_module_NAME}Targets
    )

    install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.h"
    )

    install(DIRECTORY src/
        DESTINATION include
        FILES_MATCHING PATTERN "*.hpp"
    )

    install(EXPORT aquila_${aquila_declare_module_NAME}Targets
        DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Aquila" COMPONENT dev
    )
    set(ENABLED_AQUILA_MODULES "${ENABLED_AQUILA_MODULES};${aquila_declare_module_NAME}" CACHE STRING INTERNAL FORCE)
endfunction()
