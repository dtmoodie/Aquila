find_package(cereal QUIET)

if(cereal_FOUND)
    if(NOT TARGET cereal)
        message(FATAL_ERROR "cereal_FOUND is set but no cereal target created")
    endif()
endif()

if(NOT cereal_FOUND)
    set(JUST_INSTALL_CEREAL ON CACHE BOOL "" FORCE)
    add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cereal")
endif()

