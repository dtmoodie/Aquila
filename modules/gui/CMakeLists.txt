file(GLOB_RECURSE gui_src "src/*.cpp" "src/*.hpp")

include_directories(
    ${OpenCV_INCLUDE_DIRS}
)

aquila_declare_module(NAME gui 
    SRC gui_src 
    DEPENDS 
        metaobject_object 
        metaobject_core 
        opencv_highgui
        aquila_rcc 
        metaobject_types
)

target_include_directories(aquila_gui
    PUBLIC
        $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>    
)

if(BUILD_TESTS)
    add_subdirectory(tests)
ENDIF(BUILD_TESTS)
