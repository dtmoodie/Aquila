set(MSVC_TEMPLATE_FILE_PATH "${CMAKE_CURRENT_LIST_DIR}/msvc_project_template.vcxproj.user.in" CACHE PATH INTERNAL FORCE)


macro(glob_dll_paths target outvar)
    if(TARGET ${target})
        get_target_property(type_ ${target} TYPE)
        if(${type_} STREQUAL "INTERFACE_LIBRARY")
            set(dir_property INTERFACE_LINK_DIRECTORIES)
            set(lib_property INTERFACE_LINK_LIBRARIES)
        else()
            set(dir_property LINK_DIRECTORIES)
            set(lib_property LINK_LIBRARIES)
        endif()

        get_target_property(target_link_path ${target} ${dir_property})

        if(target_link_path)
            list(APPEND ${outvar} ${target_link_path})
        endif(target_link_path)

        get_target_property(int_link_lib ${target} ${lib_property})
        
        foreach(lib ${int_link_lib})
            glob_dll_paths(${lib} ${outvar})
        endforeach(lib ${int_link_lib})

    endif(TARGET ${target})
endmacro(glob_dll_paths)

macro(setup_msvc_target target)
    if(MSVC)
        set(dll_paths "")
        glob_dll_paths(${target} dll_paths)
        list(REMOVE_DUPLICATES dll_paths)
        
        if(${MSVC_TOOLSET_VERSION} STREQUAL "140")
            set(MSVC_VERSION_STR "14.0")
        elseif(${MSVC_TOOLSET_VERSION} STREQUAL "141")
            set(MSVC_VERSION_STR "15.0")
        elseif(${MSVC_TOOLSET_VERSION} STREQUAL "142")
            set(MSVC_VERSION_STR "16.0")
        endif()

        get_target_property(bin_dir ${target} BINARY_DIR)

        configure_file("${MSVC_TEMPLATE_FILE_PATH}" ${bin_dir}/${target}.vcxproj.user @ONLY)
    endif(MSVC)
endmacro(setup_msvc_target)