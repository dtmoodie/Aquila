file(GLOB_RECURSE gui_src "src/*.cpp" "src/*.hpp")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

aquila_declare_module(NAME gui 
    SRC gui_src 
    DEPENDS 
        metaobject_object 
        metaobject_core 
        metaobject_types
        opencv_highgui

        aquila_rcc 
        aquila_core
        aquila_types

)

if(BUILD_TESTS)
    add_subdirectory(tests)
ENDIF(BUILD_TESTS)
