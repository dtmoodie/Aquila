file(GLOB_RECURSE gui_src "src/*.cpp" "src/*.hpp")

#aquila_declare_module(NAME gui SRC gui_src DEPENDS metaobject_object metaobject_core opencv_highgui aquila_rcc metaobject_types)
#RCC_TARGET_CONFIG(aquila_gui libraries_deb libraries_rel)
#if(UNIX)
#    # This is needed so that the objects inside of core are loaded into the object factory
#    IF(CMAKE_BUILD_TYPE MATCHES Debug)
#        add_custom_command(TARGET aquila_gui POST_BUILD COMMAND cd ${CMAKE_BINARY_DIR}/bin/plugins && ln -sf ../../libaquila_guid.so libaquila_guid.so)
#    else()
#        add_custom_command(TARGET aquila_gui POST_BUILD COMMAND cd ${CMAKE_BINARY_DIR}/bin/plugins && ln -sf ../../libaquila_gui.so libaquila_gui.so)
#    endif()
#    add_custom_command(TARGET aquila_gui POST_BUILD COMMAND cd ${CMAKE_BINARY_DIR}/bin/plugins && ln -sf ../../aquila_gui_config.txt aquila_gui_config.txt)
#endif()
