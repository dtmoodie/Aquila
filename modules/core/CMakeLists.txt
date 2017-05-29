file(GLOB_RECURSE src "src/*.cpp" "src/*.cu" "src/*.hpp")

aquila_declare_module(NAME core SRC src
    DEPENDS
        metaobject_core 
		metaobject_object 
		metaobject_serialization
        metaobject_params 
		aquila_rcc 
		aquila_gui
		metaobject_metaparams
)

# privately need headers from these modules
target_include_directories(aquila_core
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../utilities/src
)
target_include_directories(aquila_core
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../serialization/src
)
