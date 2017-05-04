if(MSVC)

else()
    aquila_declare_module(NAME types DEPENDS opencv_core MetaObject FLAGS "-std=c++11")
endif()
