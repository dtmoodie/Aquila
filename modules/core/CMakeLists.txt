file(GLOB_RECURSE src "src/*.cpp" "src/*.cu" "src/*.hpp")

aquila_declare_module(NAME core SRC src
    DEPENDS
        metaobject_core
        metaobject_object
        metaobject_serialization
        metaobject_params
        aquila_gui
        aquila_rcc
        metaobject_metaparams
)

# privately need headers from these modules
target_include_directories(aquila_core
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../utilities/src
)
target_include_directories(aquila_core
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../serialization/src
)
IF(WIN32)
set_target_properties(aquila_core
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo
)
ENDIF()

RCC_TARGET_CONFIG(aquila_core libraries_deb libraries_rel)

if(UNIX)
    # This is needed so that the objects inside of core are loaded into the object factory
    IF(CMAKE_BUILD_TYPE MATCHES Debug)
        add_custom_command(TARGET aquila_core POST_BUILD COMMAND cd ${CMAKE_BINARY_DIR}/bin/Plugins && ln -sf ../../libaquila_cored.so libaquila_cored.so)
    else()
        add_custom_command(TARGET aquila_core POST_BUILD COMMAND cd ${CMAKE_BINARY_DIR}/bin/Plugins && ln -sf ../../libaquila_core.so libaquila_core.so)
    endif()
    add_custom_command(TARGET aquila_core POST_BUILD COMMAND cd ${CMAKE_BINARY_DIR}/bin/Plugins && ln -sf ../../aquila_core_config.txt aquila_core_config.txt)
endif()
