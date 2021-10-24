set(plugin_config_script "${CMAKE_CURRENT_LIST_DIR}/parse_cmake.py" CACHE STRING "" FORCE)
set(plugin_export_template_path "${CMAKE_CURRENT_LIST_DIR}/plugin_export.hpp.in" CACHE STRING "" FORCE)
set(plugin_config_file_path "${CMAKE_CURRENT_LIST_DIR}/plugin_export.cpp.in" CACHE STRING "" FORCE)
set(plugin_link_lib_input_path "${CMAKE_CURRENT_LIST_DIR}/link_libs.hpp.in" CACHE STRING "" FORCE)

macro(metaobject_declare_plugin tgt)
    set(options NOINSTALL)
    cmake_parse_arguments(metaobject_declare_plugin "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )
    macro(RCC_HANDLE_LIB TARGET)
      if(RCC_VERBOSE_CONFIG)
        message(STATUS "===================================================================\n"
                       "  RCC config information for ${TARGET}")
      endif(RCC_VERBOSE_CONFIG)
      foreach(lib ${ARGN})
      endforeach(lib ${ARGN})
    endmacro(RCC_HANDLE_LIB target lib)

    get_target_property(target_include_dirs_ ${tgt} INCLUDE_DIRECTORIES)
    get_target_property(target_link_libs_    ${tgt} LINK_LIBRARIES)
    set_property(TARGET ${tgt} APPEND PROPERTY RCC_MODULE)
    set_target_properties(${tgt}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
            ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/plugins
            FOLDER plugins
            RCC_MODULE ON
    )

    target_include_directories(${tgt}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/src/>
            $<INSTALL_INTERFACE:include>
    )
    target_include_directories(${tgt}
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/plugins/>
            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/plugins/${tgt}/>
    )

    RCC_TARGET_CONFIG(${tgt} plugin_libraries plugin_libraries_debug plugin_libraries_release)

    if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/${tgt}_config.txt")
      FILE(READ "${CMAKE_CURRENT_BINARY_DIR}/${tgt}_config.txt" temp)
    endif()

    SET(PROJECT_ID)
    IF(temp)
      STRING(FIND "${temp}" "\n" len)
      STRING(SUBSTRING "${temp}" 0 ${len} temp)
      SET(PROJECT_ID ${temp})
      if(RCC_VERBOSE_CONFIG)
        message("Project ID for ${tgt}: ${PROJECT_ID}")
      endif()
    ELSE(temp)
      SET(PROJECT_ID "-1")
    ENDIF(temp)

    SET(LINK_LIBS ${plugin_libraries})
    set(LINK_LIBS_RELEASE ${plugin_libraries_release})
    set(LINK_LIBS_DEBUG ${plugin_libraries_debug})

    set(${tgt}_PLUGIN_INCLUDE_DIRS "${CMAKE_CURRENT_LIST_DIR}/src/" CACHE PATH "" FORCE)


    set(PLUGIN_NAME ${tgt})
    string(TIMESTAMP BUILD_DATE "%Y-%m-%d %H:%M")

    CONFIGURE_FILE(${plugin_export_template_path} "${CMAKE_BINARY_DIR}/plugins/${tgt}/${tgt}_export.hpp" @ONLY)
    CONFIGURE_FILE(${plugin_link_lib_input_path} "${CMAKE_BINARY_DIR}/plugins/${tgt}/${tgt}_link_libs.hpp" @ONLY)

    CONFIGURE_FILE("${plugin_config_file_path}" "${CMAKE_BINARY_DIR}/plugins/${tgt}/plugin_config.cpp" @ONLY)

    FILE(APPEND "${CMAKE_BINARY_DIR}/bin/plugins/${tgt}_config.txt" "\n\nlink_libs:\n")
    if(NOT WIN32)
        FILE(APPEND "${CMAKE_BINARY_DIR}/bin/plugins/${tgt}_config.txt" "pthread\n")
    endif()


    foreach(lib ${LINK_LIBS})
        FILE(APPEND "${CMAKE_BINARY_DIR}/bin/plugins/${tgt}_config.txt" "${lib}\n")
    endforeach()
    FILE(APPEND "${CMAKE_BINARY_DIR}/bin/plugins/${tgt}_config.txt" "\n\nlink_libs_debug:\n")
    foreach(lib ${LINK_LIBS_DEBUG})
        FILE(APPEND "${CMAKE_BINARY_DIR}/bin/plugins/${tgt}_config.txt" "${lib}\n")
    endforeach()
    FILE(APPEND "${CMAKE_BINARY_DIR}/bin/plugins/${tgt}_config.txt" "\n\nlink_libs_release:\n")
    foreach(lib ${LINK_LIBS_RELEASE})
        FILE(APPEND "${CMAKE_BINARY_DIR}/bin/plugins/${tgt}_config.txt" "${lib}\n")
    endforeach()

    execute_process(
        COMMAND
            python
            ${plugin_config_script}
            --in_path ${CMAKE_BINARY_DIR}/bin/plugins/${tgt}_config.txt
            --out_path ${CMAKE_BINARY_DIR}/plugins/${tgt}/plugin_config2.cpp
            --plugin_name ${tgt}
            --link_path ${CMAKE_BINARY_DIR}/plugins/${tgt}/${tgt}_link_libs.hpp
    )

    set_property(TARGET ${tgt} APPEND PROPERTY SOURCES "${CMAKE_BINARY_DIR}/plugins/${tgt}/plugin_config.cpp")
    set_property(TARGET ${tgt} APPEND PROPERTY SOURCES "${CMAKE_BINARY_DIR}/plugins/${tgt}/plugin_config2.cpp")
    set_property(TARGET ${tgt} APPEND PROPERTY SOURCES "${CMAKE_BINARY_DIR}/plugins/${tgt}/${tgt}_export.hpp")
    set_property(TARGET ${tgt} APPEND PROPERTY SOURCES "${CMAKE_BINARY_DIR}/plugins/${tgt}/${tgt}_link_libs.hpp")

    LINK_DIRECTORIES(${LINK_DIRS_DEBUG})
    LINK_DIRECTORIES(${LINK_DIRS_RELEASE})
    LINK_DIRECTORIES(${LINK_DIRS})

    # ============= Write out a file containing external include info

    set(external_include_file "#pragma once\n\n#include \"RuntimeObjectSystem/RuntimeLinkLibrary.h\"\n\n")

    # wndows link libs
    if(LINK_LIBS_RELEASE)
      LIST(REMOVE_DUPLICATES LINK_LIBS_RELEASE)
          list(SORT LINK_LIBS_RELEASE)
    endif()
    if(LINK_LIBS_DEBUG)
      LIST(REMOVE_DUPLICATES LINK_LIBS_DEBUG)
          list(SORT LINK_LIBS_DEBUG)
    endif()
    if(NOT WIN32)
        set(external_include_file "${external_include_file}\nRUNTIME_COMPILER_LINKLIBRARY(\"-lpthread\")\n")
    endif()
    set(external_include_file "${external_include_file}\n#if defined(NDEBUG) && !defined(_DEBUG)\n\n")
    if(WIN32)
        set(prefix "")
        set(postfix ".lib")
    else(WIN32)
        set(prefix "-l")
        set(postfix "")
    endif(WIN32)

    foreach(lib ${LINK_LIBS_RELEASE})
        string(LENGTH ${lib} len)
        if(len GREATER 3)
            string(SUBSTRING "${lib}" 0 3 sub)
            if(${sub} STREQUAL lib)
              string(SUBSTRING "${lib}" 3 -1 lib)
              set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
            else()
              set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
            endif()
        else()
            set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
        endif()
    endforeach()

    set(external_include_file "${external_include_file}\n  #else\n\n")

    foreach(lib ${LINK_LIBS_DEBUG})
        string(LENGTH ${lib} len)
        if(len GREATER 3)
            string(SUBSTRING "${lib}" 0 3 sub)
            if(${sub} STREQUAL lib)
                string(SUBSTRING "${lib}" 3 -1 lib)
                set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
            else()
                set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
            endif()
        else()
            set(external_include_file "${external_include_file}    RUNTIME_COMPILER_LINKLIBRARY(\"${prefix}${lib}${postfix}\")\n")
        endif()
    endforeach()

    set(external_include_file "${external_include_file}\n  #endif // NDEBUG\n")

    set(external_include_file "${external_include_file}\n")

    if(NOT ${metaobject_declare_plugin_NOINSTALL})
      INSTALL(TARGETS ${tgt}
              LIBRARY DESTINATION bin/plugins
              RUNTIME DESTINATION bin
      )

      export(TARGETS ${tgt} FILE "${PROJECT_BINARY_DIR}/${tgt}Targets.cmake")
      export(PACKAGE ${tgt})

      install(TARGETS ${tgt}
          DESTINATION lib
          EXPORT ${tgt}Targets
      )

      INSTALL(DIRECTORY src/ DESTINATION include/${tgt} FILES_MATCHING PATTERN "*.hpp")
      INSTALL(DIRECTORY src/ DESTINATION include/${tgt} FILES_MATCHING PATTERN "*.h")
      install(EXPORT ${tgt}Targets DESTINATION "${CMAKE_INSTALL_PREFIX}/share/metaobject" COMPONENT dev)
    endif(NOT ${metaobject_declare_plugin_NOINSTALL})

    IF(RCC_VERBOSE_CONFIG)
      string(REGEX REPLACE ";" "\n    " include_dirs_ "${INCLUDE_DIRS}")
      string(REGEX REPLACE ";" "\n    " link_dirs_release_ "${LINK_DIRS_RELEASE}")
      string(REGEX REPLACE ";" "\n    " link_dirs_debug_ "${LINK_DIRS_DEBUG}")
      MESSAGE(STATUS
      "  ${outfile_}
      Include Dirs:
        ${include_dirs_}
      Link Dirs Debug:
        ${link_dirs_debug_}
      Link Dirs Release:
        ${link_dirs_release_}
      Link libs Release:
        ${LINK_LIBS_RELEASE}
      Link libs Debug:
        ${LINK_LIBS_DEBUG}
     ")
    ENDIF(RCC_VERBOSE_CONFIG)

endmacro(metaobject_declare_plugin)


function(metaobject_install_dependent_lib FILENAME)

    if(EXISTS ${FILENAME})
        if(IS_SYMLINK ${FILENAME})
            get_filename_component(path ${FILENAME} REALPATH)
            INSTALL(FILES ${path} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
            INSTALL(FILES ${FILENAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
        else()
            INSTALL(FILES ${FILENAME} DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
        endif()
    endif()

endfunction(metaobject_install_dependent_lib)
