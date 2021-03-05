
aquila_declare_module(NAME serialization
    DEPENDS
        aquila_types
        aquila_core
        metaobject_params
)

if(TARGET metaobject_python)
  include_directories(${PYTHON_INCLUDE_DIR})
  target_link_libraries(aquila_serialization PUBLIC metaobject_python)
  target_compile_definitions(aquila_serialization PRIVATE "MO_HAVE_PYTHON")
endif(TARGET metaobject_python)