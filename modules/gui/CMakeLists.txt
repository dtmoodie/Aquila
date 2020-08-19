file(GLOB_RECURSE gui_src "src/*.cpp" "src/*.hpp")

aquila_declare_module(NAME gui 
    SRC gui_src 
    DEPENDS 
        metaobject_object 
        metaobject_core 
        opencv_highgui 
        aquila_rcc 
        metaobject_types
)

if(BUILD_TESTS)
    add_subdirectory(tests)
ENDIF(BUILD_TESTS)
