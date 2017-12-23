function(aquila_declare_module)
    set(oneValueArgs NAME CXX_STANDARD INCLUDE)
    set(multiValueArgs SRC DEPENDS FLAGS PRIVATE_DEP_HEADER)
    cmake_parse_arguments(aquila_declare_module "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    if(${aquila_declare_module_SRC})
        cuda_add_library(aquila_${aquila_declare_module_NAME} SHARED ${${aquila_declare_module_SRC}})
    else()
        file(GLOB_RECURSE src "src/*.cpp" "src/*.h" "src/*.cu" "src/*.hpp")
        cuda_add_library(aquila_${aquila_declare_module_NAME} SHARED ${src})
    endif()
    set(aquila_modules "${aquila_modules};aquila_${aquila_declare_module_NAME}" CACHE INTERNAL "" FORCE)
    set(aquila_module_includes "${aquila_module_includes};${CMAKE_CURRENT_LIST_DIR}/src" CACHE INTERNAL "" FORCE)
    if(aquila_declare_module_DEPENDS)
        rcc_link_lib(aquila_${aquila_declare_module_NAME} ${aquila_declare_module_DEPENDS})
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
    target_include_directories(aquila_${aquila_declare_module_NAME}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src>
            $<INSTALL_INTERFACE:include>
    )
    set_target_properties(aquila_${aquila_declare_module_NAME}
        PROPERTIES
            CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    if(aquila_declare_module_CXX_STANDARD)
        set_target_properties(aquila_${aquila_declare_module_NAME} PROPERTIES CXX_STANDARD ${aquila_declare_module_CXX_STANDARD})
    endif()

    if(aquila_declare_module_FLAGS)
        target_compile_options(aquila_${aquila_declare_module_NAME} PUBLIC ${aquila_declare_module_FLAGS})
    endif()

    export(TARGETS aquila_${aquila_declare_module_NAME}
        FILE "${PROJECT_BINARY_DIR}/AquilaTargets-${aquila_declare_module_NAME}.cmake"
    )

    install(TARGETS aquila_${aquila_declare_module_NAME}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )

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

    set_target_properties(aquila_${aquila_declare_module_NAME} PROPERTIES FOLDER Aquila)
endfunction()
