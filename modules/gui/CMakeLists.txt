file(GLOB_RECURSE gui_src "src/Aquila/gui/*.cpp" "src/Aquila/gui/*.hpp")

aquila_declare_module(NAME gui SRC gui_src DEPENDS metaobject_object aquila_rcc metaobject_core)
