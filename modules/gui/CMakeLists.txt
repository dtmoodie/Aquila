file(GLOB_RECURSE gui_src "src/Aquila/gui/*.cpp" "src/Aquila/gui/*.hpp")

aquila_declare_module(NAME gui SRC gui_src DEPENDS metaobject_object aquila_rcc metaobject_core opencv_highgui)
if(UNIX)
    # This is needed so that the objects inside of core are loaded into the object factory
    IF(CMAKE_BUILD_TYPE MATCHES Debug)
        add_custom_command(TARGET aquila_gui POST_BUILD COMMAND cd ${CMAKE_BINARY_DIR}/bin/Plugins && ln -sf ../../libaquila_guid.so libaquila_guid.so)
    else()
        add_custom_command(TARGET aquila_gui POST_BUILD COMMAND cd ${CMAKE_BINARY_DIR}/bin/Plugins && ln -sf ../../libaquila_gui.so libaquila_gui.so)
    endif()
endif()
