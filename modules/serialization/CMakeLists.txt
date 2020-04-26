  include_directories(${PYTHON_INCLUDE_DIR})

aquila_declare_module(NAME serialization
    DEPENDS
        aquila_types
        metaobject_params
        aquila_core
)
