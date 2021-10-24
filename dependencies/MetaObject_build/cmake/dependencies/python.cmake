if(WITH_PYTHON)
    find_package(Python COMPONENTS Interpreter Development NumPy)

    if(Python_FOUND)
        if(${Python_VERSION} VERSION_GREATER_EQUAL 3.0.0)
            include(${CMAKE_CURRENT_LIST_DIR}/python3.cmake)
        else(${Python_VERSION} VERSION_GREATER_EQUAL 3.0.0)
            include(${CMAKE_CURRENT_LIST_DIR}/python2.cmake)
        endif(${Python_VERSION} VERSION_GREATER_EQUAL 3.0.0)
    endif(Python_FOUND)
    
else(WITH_PYTHON)
    set(MO_PYTHON_STATUS "Python disabled by WITH_PYTHON cmake flag")
endif(WITH_PYTHON)
