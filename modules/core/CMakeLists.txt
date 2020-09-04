file(GLOB_RECURSE src "src/*.cpp" "src/*.cu" "src/*.hpp")

aquila_declare_module(NAME core SRC src
    DEPENDS
        metaobject_core
        metaobject_object
        metaobject_serialization
        metaobject_params
        aquila_rcc
        aquila_utilities
        metaobject_metaparams
        ce
)

# privately need headers from these modules
target_include_directories(aquila_core
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../utilities/src
)

target_include_directories(aquila_core
    PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../serialization/src
)

IF(WIN32)
set_target_properties(aquila_core
    PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/Release
        ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/Debug
        ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/RelWithDebInfo
)
ENDIF()

RCC_TARGET_CONFIG(aquila_core libraries_deb libraries_rel)


SUBDIRLIST(tests "${CMAKE_CURRENT_LIST_DIR}/tests")
foreach(test ${tests})
    file(GLOB_RECURSE test_hdr "tests/${test}/*.hpp")
    file(GLOB_RECURSE test_srcs "tests/${test}/*.cpp")
    file(GLOB_RECURSE test_knl "tests/${test}/*.cu")
    LIST(LENGTH test_knl num_knl)
    set(test_name test_aq_${test})
    IF(${num_knl} GREATER 0)
        cuda_add_executable(${test_name} ${test_srcs} ${test_knl} ${test_hdr})
    else()
        add_executable(${test_name} ${test_srcs}  ${test_hdr})
    endif()
    TARGET_LINK_LIBRARIES(${test_name} ${MetaObject_LIBRARIES}  aquila_core gtest gtest_main)
    set_target_properties(${test_name} PROPERTIES FOLDER Tests/Aquila)
    add_test(${test_name} ${test_name})
    if(MSVC)
      CONFIGURE_FILE("tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/Aquila/${test}.vcxproj.user @ONLY)
      CONFIGURE_FILE("tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/${test}.vcxproj.user @ONLY)
    endif()
endforeach()
