if(MSVC)

else()
    aquila_declare_module(NAME types DEPENDS opencv_core MetaObject CXX_STANDARD 14)
endif()
