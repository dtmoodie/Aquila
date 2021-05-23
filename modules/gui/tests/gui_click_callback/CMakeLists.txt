add_executable(test_aq_gui_click_callback main.cpp)

target_link_libraries(test_aq_gui_click_callback 
    aquila_gui
    aquila_types
    metaobject_runtime_reflection
    metaobject_core
    ce
)

