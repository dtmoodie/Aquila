if(TARGET metaobject_python)
    file(GLOB_RECURSE src "src/*.cpp" "src/*.cu" "src/*.hpp")
    aquila_declare_module(NAME python SRC src DEPENDS
            metaobject_core
            metaobject_object
            metaobject_serialization
            metaobject_params
            metaobject_metaparams
            metaobject_python

            aquila_rcc
            aquila_types
            aquila_serialization
    )

    if(TARGET aquila_gui)
        target_link_libraries(aquila_python
            aquila_gui
        )
        target_compile_definitions(aquila_python
            PRIVATE
                HAVE_AQUILA_GUI
        )
    endif()

    if(UNIX)
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            set(POSTFIX "${CMAKE_DEBUG_POSTFIX}")
        endif()
        get_target_property(out_dir aquila_python ARCHIVE_OUTPUT_DIRECTORY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/aquila.py.in.unix ${out_dir}/aquila.py @ONLY)
        add_custom_target(aquila.py.in.unix ALL)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/tests/aquila_python_test.py ${out_dir}/aquila_python_test.py @ONLY)
    else()
        get_target_property(out_dir aquila_python ARCHIVE_OUTPUT_DIRECTORY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/aquila.py.in.win32 ${out_dir}/Debug/aquila.py @ONLY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/aquila.py.in.win32 ${out_dir}/RelWithDebInfo/aquila.py @ONLY)
        configure_file(${CMAKE_CURRENT_LIST_DIR}/aquila.py.in.win32 ${out_dir}/Release/aquila.py @ONLY)
    endif(UNIX)

    if(Python_LIBRARY_RELEASE)
        if(EXISTS ${Python_LIBRARY_RELEASE})
            target_link_libraries(aquila_python
                ${Python_LIBRARY_RELEASE}
            )
        endif()
    endif()

    RCC_TARGET_CONFIG(aquila_python libraries_deb libraries_rel)
    INSTALL(FILES ${out_dir}/aquila.py
        DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    )

endif(TARGET metaobject_python)
