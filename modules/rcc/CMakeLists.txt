file(GLOB_RECURSE src "src/*.hpp" "src/*.cpp")

aquila_declare_module(NAME rcc SRC src DEPENDS RuntimeObjectSystem RuntimeCompiler metaobject_core)
