# helper function to recursively get dependent library directories and includes

macro(_handle_imported_target LIB_DIR_VAR LIB_FILES LIB_FILES_DEBUG LIB_FILES_RELEASE TGT tab)
    if(NOT "${TGT}" STREQUAL Threads::Threads)
        set(found_ OFF)
        if(NOT WIN32)
            get_target_property(tgt_location ${TGT} IMPORTED_IMPLIB)
            if(tgt_location)
                set(found_ ON)
                get_filename_component(dir_ ${tgt_location} DIRECTORY)
                list(APPEND ${LIB_DIR_VAR} ${dir_})
                set(tgt_location "")
            else(tgt_location)
                get_target_property(tgt_location ${TGT} LOCATION)
                if(tgt_location)
                    set(found_ ON)
                    #rcc_strip_extension(tgt_location name_)
                    #rcc_stip_lib(${name_} name_)
                    get_filename_component(dir_ ${tgt_location} DIRECTORY)
                    LIST(APPEND ${LIB_DIR_VAR} ${dir_})
                    LIST(APPEND ${LIB_FILES} ${tgt_location})
                    set(tgt_location "")
                endif(tgt_location)
            endif(tgt_location)
        endif(NOT WIN32)

        get_target_property(tgt_location ${TGT} IMPORTED_IMPLIB_RELEASE)
        if(tgt_location)
            set(found_ ON)
            LIST(APPEND ${LIB_DIR_VAR} ${dir_})
            LIST(APPEND ${LIB_FILES_RELEASE} ${tgt_location})
            set(tgt_location "")
        else()
            get_target_property(tgt_location ${TGT} LOCATION_RELEASE)
            if(tgt_location)
                set(found_ ON)
                get_filename_component(dir_ ${tgt_location} DIRECTORY)
                list(APPEND ${LIB_DIR_VAR} ${dir_})
                LIST(APPEND ${LIB_FILES_RELEASE} ${tgt_location})
                set(tgt_location "")
            endif()
        endif()

        get_target_property(tgt_location ${TGT} IMPORTED_IMPLIB_DEBUG)
        if(tgt_location)
            get_filename_component(dir_ ${tgt_location} DIRECTORY)
            list(APPEND ${LIB_DIR_VAR} ${dir_})
            LIST(APPEND ${LIB_FILES_DEBUG} ${tgt_location})
            set(tgt_location "")
            if(NOT found_)
                LIST(APPEND ${LIB_FILES_RELEASE} ${name_})
            endif()
        else()
            get_target_property(tgt_location ${TGT} LOCATION_DEBUG)
            if(tgt_location)
                get_filename_component(dir_ ${tgt_location} DIRECTORY)
                list(APPEND ${LIB_DIR_VAR} ${dir_})
                LIST(APPEND ${LIB_FILES_DEBUG} ${tgt_location})
                if(NOT found_)
                    LIST(APPEND ${LIB_FILES_RELEASE} ${tgt_location})
                endif()
                set(tgt_location "")
            endif()
        endif()
    endif(NOT "${TGT}" STREQUAL Threads::Threads)
endmacro(_handle_imported_target)

macro(__target_helper LIB_DIR_VAR INC_VAR LIB_FILES LIB_FILES_DEBUG LIB_FILES_RELEASE DEPS FLAGS DEFS tgt tab TGTS)
    list(FIND ${TGTS} ${tgt} index_)
    if(${index_} EQUAL -1 AND NOT "${tgt}" STREQUAL Threads::Threads)
        list(APPEND ${TGTS} ${tgt})
        if(TARGET ${tgt})
            get_target_property(imported_ ${tgt} IMPORTED)
            get_target_property(type_ ${tgt} TYPE)
            if(${type_} STREQUAL "INTERFACE_LIBRARY")
                get_target_property(inc_dir ${tgt} INTERFACE_INCLUDE_DIRECTORIES)
                if(inc_dir)
                    list(APPEND ${INC_VAR} ${inc_dir})
                endif()

                get_target_property(compile_features ${tgt} INTERFACE_COMPILE_FEATURES)
                # TODO things and stuff
                if(compile_features)
                    if("cxx_return_type_deduction" STREQUAL ${compile_features})
                        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
                            if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.9)
                                LIST(APPEND ${FLAGS} "-std=c++14")
                            else()
                                LIST(APPEND ${FLAGS} "-std=c++1y")
                            endif()
                        endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
                    endif()
                endif()

                get_target_property(compile_flags ${tgt} INTERFACE_COMPILE_OPTIONS)
                if(compile_flags)
                    LIST(APPEND ${FLAGS} "${compile_flags}")
                endif(compile_flags)

                get_target_property(compile_definitions ${tgt} INTERFACE_COMPILE_DEFINITIONS)
                if(compile_definitions)
                    LIST(APPEND ${DEFS} "${compile_definitions}")
                endif(compile_definitions)

                get_target_property(compile_definitions ${tgt} INTERFACE_COMPILE_DEFINITIONS_RELEASE)
                if(compile_definitions)
                    LIST(APPEND ${DEFS} "${compile_definitions}")
                endif(compile_definitions)

                get_target_property(compile_definitions ${tgt} INTERFACE_COMPILE_DEFINITIONS_DEBUG)
                if(compile_definitions)
                    LIST(APPEND ${DEFS} "${compile_definitions}")
                endif(compile_definitions)


                get_target_property(int_link_lib ${tgt} INTERFACE_LINK_LIBRARIES)
                foreach(lib ${int_link_lib})
                    if(TARGET ${lib})
                        if(RCC_VERBOSE_CONFIG)
                            message("${tab} Adding ${lib} to ${LIB_FILES_DEBUG}")
                        endif()
                        __target_helper(${LIB_DIR_VAR}
                            ${INC_VAR}
                            ${LIB_FILES}
                            ${LIB_FILES_DEBUG}
                            ${LIB_FILES_RELEASE}
                            ${DEPS}
                            ${FLAGS}
                            ${DEFS}
                            ${lib}
                            "${tab}  "
                            ${TGTS}
                        )
                    else(TARGET ${lib})
                        if(EXISTS ${lib})
                            get_filename_component(dir_ ${lib} DIRECTORY)
                            rcc_strip_extension(lib name_)
                            rcc_stip_lib(${name_} name_)
                            list(APPEND ${LIB_DIR_VAR} ${dir_})
                            list(APPEND ${LIB_FILES} ${lib})
                        endif(EXISTS ${lib})
                    endif(TARGET ${lib})
                endforeach(lib ${int_link_lib})

            else(${type_} STREQUAL "INTERFACE_LIBRARY")
                get_target_property(rcc_mod ${tgt} RCC_MODULE)
                if(rcc_mod)
                    list(APPEND ${DEPS} "${tgt}")
                endif()

                if(${imported_})
                    _handle_imported_target(${LIB_DIR_VAR} ${LIB_FILES} ${LIB_FILES_DEBUG} ${LIB_FILES_RELEASE} ${tgt} "${tab}  ")
                else(${imported_})
                    get_target_property(out_name ${tgt} NAME)
                    get_target_property(debug_postfix ${tgt} DEBUG_POSTFIX)
                    if(out_name)
                        list(APPEND ${LIB_FILES_DEBUG} ${out_name}${debug_postfix})
                        list(APPEND ${LIB_FILES_RELEASE} ${out_name})
                    endif(out_name)
                    get_target_property(out_dir ${tgt} LIBRARY_OUTPUT_DIRECTORY)
                    if(out_dir)
                        list(APPEND ${LIB_DIR_VAR} ${out_dir})
                    endif(out_dir)
                    get_target_property(lib_path ${tgt} LIBRARY_OUTPUT_DIRECTORY)
                    if(lib_path)
                        list(APPEND ${LIB_DIR_VAR} ${out_dir})
                    endif(lib_path)
                    get_target_property(inc_dir ${tgt} INTERFACE_INCLUDE_DIRECTORIES)
                    if(inc_dir)
                        list(APPEND ${INC_VAR} ${inc_dir})
                    endif()
                    get_target_property(inc_dir ${tgt} INCLUDE_DIRECTORIES)
                    if(inc_dir)
                        list(APPEND ${INC_VAR} ${inc_dir})
                    endif()
                    get_target_property(defs ${tgt} COMPILE_DEFINITIONS)
                    if(defs)
                        list(APPEND ${DEFS} ${defs})
                    endif()

                    get_target_property(defs ${tgt} COMPILE_DEFINITIONS_RELEASE)
                    if(defs)
                        list(APPEND ${DEFS} ${defs})
                    endif()

                    get_target_property(defs ${tgt} COMPILE_DEFINITIONS_DEBUG)
                    if(defs)
                        list(APPEND ${DEFS} ${defs})
                    endif()

                    get_target_property(compile_definitions ${tgt} INTERFACE_COMPILE_DEFINITIONS)
                    if(compile_definitions)
                        LIST(APPEND ${DEFS} "${compile_definitions}")
                    endif(compile_definitions)

                    get_target_property(compile_definitions ${tgt} INTERFACE_COMPILE_DEFINITIONS_RELEASE)
                    if(compile_definitions)
                        LIST(APPEND ${DEFS} "${compile_definitions}")
                    endif(compile_definitions)

                    get_target_property(compile_definitions ${tgt} INTERFACE_COMPILE_DEFINITIONS_DEBUG)
                    if(compile_definitions)
                        LIST(APPEND ${DEFS} "${compile_definitions}")
                    endif(compile_definitions)

                    if(RCC_VERBOSE_CONFIG)
                        message(STATUS "${tab}${tgt} (${inc_dir}) (${lib_path})")
                    endif(RCC_VERBOSE_CONFIG)
                endif(${imported_})

                get_target_property(int_link_lib ${tgt} INTERFACE_LINK_LIBRARIES)
                foreach(lib ${int_link_lib})
                    if(TARGET ${lib})
                        if(RCC_VERBOSE_CONFIG)
                            message("${tab} Adding ${lib} to ${LIB_FILES_DEBUG}")
                        endif()
                        __target_helper(${LIB_DIR_VAR} ${INC_VAR} ${LIB_FILES} ${LIB_FILES_DEBUG} ${LIB_FILES_RELEASE}  ${DEPS} ${FLAGS} ${DEFS} ${lib} "${tab}  " ${TGTS})
                    else(TARGET ${lib})
                        if(EXISTS ${lib})
                            get_filename_component(dir_ ${lib} DIRECTORY)
                            list(APPEND ${LIB_DIR_VAR} ${dir_})
                            rcc_strip_extension(lib name_)                            
                            rcc_stip_lib(${name_} name_)
                            list(APPEND ${LIB_FILES_DEBUG} ${name_})
                        endif(EXISTS ${lib})
                    endif(TARGET ${lib})
                endforeach(lib ${int_link_lib})

                get_target_property(int_link_lib ${tgt} LINK_LIBRARIES)
                foreach(lib ${int_link_lib})

                    if(TARGET ${lib})
                        __target_helper(${LIB_DIR_VAR} ${INC_VAR} ${LIB_FILES} ${LIB_FILES_DEBUG} ${LIB_FILES_RELEASE} ${DEPS} ${FLAGS} ${DEFS} ${lib} "${tab}  " ${TGTS})
                    else(TARGET ${lib})
                        if(EXISTS ${lib})
                            get_filename_component(dir_ ${lib} DIRECTORY)
                            list(APPEND ${LIB_DIR_VAR} ${dir_})
                            rcc_strip_extension(lib name_)
                            rcc_stip_lib(${name_} name_)
                            list(APPEND ${LIB_FILES_DEBUG} ${name_})
                            # TODO VALIDATE
                            list(APPEND ${LIB_FILES_RELEASE} ${name_})
                        endif(EXISTS ${lib})
                    endif(TARGET ${lib})
                endforeach(lib ${int_link_lib})
            endif(${type_} STREQUAL "INTERFACE_LIBRARY")
        else(TARGET ${tgt})
            # not a target but a file
            if(EXISTS ${tgt})
                get_filename_component(dir_ ${tgt} DIRECTORY)
                list(APPEND ${LIB_DIR_VAR} ${dir_})
                rcc_strip_extension(tgt name_)
                list(APPEND ${LIB_FILES_DEBUG} ${name_})
                # TODO VALIDATE
                list(APPEND ${LIB_FILES_RELEASE} ${name_})
            endif(EXISTS ${tgt})
        endif(TARGET ${tgt})
    endif(${index_} EQUAL -1 AND NOT "${tgt}" STREQUAL Threads::Threads)
endmacro(__target_helper)

macro(_target_helper LIB_DIR_VAR INC_VAR LIB_FILES LIB_FILES_DEBUG LIB_FILES_RELEASE DEPS COMPILE_FLAGS COMPILE_DEFINITIONS tgt tab)
    set(TGT_LIST "")
    __target_helper(${LIB_DIR_VAR} ${INC_VAR} ${LIB_FILES} ${LIB_FILES_DEBUG} ${LIB_FILES_RELEASE} ${DEPS} ${COMPILE_FLAGS} ${COMPILE_DEFINITIONS} ${tgt} ${tab} TGT_LIST)
endmacro(_target_helper)

macro(RCC_TARGET_CONFIG target LIB_FILES_VAR LIB_FILES_DEBUG_VAR LIB_FILES_RELEASE_VAR)
    set(inc_dirs "")
    set(lib_dirs "")
    set(${LIB_FILES_DEBUG_VAR} "")
    set(${LIB_FILES_RELEASE_VAR} "")
    set(flags "")
    set(DEFS "")
    set(deps "")
    _target_helper(lib_dirs inc_dirs ${LIB_FILES_VAR} ${LIB_FILES_DEBUG_VAR} ${LIB_FILES_RELEASE_VAR} deps flags DEFS ${target} "  ")

    get_target_property(dest_dir ${target} CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    if(NOT dest_dir)
        get_target_property(dest_dir ${target} LIBRARY_OUTPUT_DIRECTORY)
    endif()
        list(APPEND lib_dirs ${dest_dir})
    if(MSVC)
        get_target_property(dest_dir_deb ${target} ARCHIVE_OUTPUT_DIRECTORY_DEBUG)
        if(NOT dest_dir_deb)
            get_target_property(dest_dir_deb ${target} ARCHIVE_OUTPUT_DIRECTORY)
            set(dest_dir_deb ${dest_dir_deb}/Debug)
        endif()
        get_target_property(dest_dir_rel ${target} ARCHIVE_OUTPUT_DIRECTORY_RELEASE)
        if(NOT dest_dir_rel)
            get_target_property(dest_dir_rel ${target} ARCHIVE_OUTPUT_DIRECTORY)
            set(dest_dir_rel ${dest_dir_rel}/Release)
        endif()
        get_target_property(dest_dir_reldeb ${target} ARCHIVE_OUTPUT_DIRECTORY_RELWITHDEBINFO)
        if(NOT dest_dir_reldeb)
            get_target_property(dest_dir_reldeb ${target} ARCHIVE_OUTPUT_DIRECTORY)
            set(dest_dir_reldeb ${dest_dir_reldeb}/RelWithDebInfo)
        endif()
        list(APPEND lib_dirs ${dest_dir_deb})
        list(APPEND lib_dirs ${dest_dir_rel})
        list(APPEND lib_dirs ${dest_dir_reldeb})
    endif(MSVC)

    get_target_property(flags_ ${target} COMPILE_OPTIONS)
    if(flags_)
        set(flags ${flags_})
    endif()
    get_directory_property(flags_ ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_OPTIONS)
    if(flags_)
        set(flags ${flags_})
    endif()
    list(APPEND flags ${CMAKE_CXX_FLAGS})

    get_target_property(defs_ ${target} COMPILE_DEFINITIONS)
    if(defs_)
        set(defs "${DEFS};${defs_}")
    endif()

    get_target_property(defs_ ${target} COMPILE_DEFINITIONS_DEBUG)
    if(defs_)
        set(defs "${DEFS};${defs_}")
    endif()

    get_target_property(defs_ ${target} COMPILE_DEFINITIONS_RELEASE)
    if(defs_)
        set(defs "${DEFS};${defs_}")
    endif()

    get_directory_property(defs_ DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_DEFINITIONS)
    if(defs_)
        set(defs "${DEFS};${defs_}")
    endif()
    get_directory_property(link_dirs ${CMAKE_CURRENT_SOURCE_DIR} LINK_DIRECTORIES)
    if(link_dirs)
            list(APPEND lib_dirs ${link_dirs})
    endif(link_dirs)

    set(NVCC_COMPILER "")
    IF(CUDA_FOUND)
        set(NVCC_COMPILER "${CUDA_NVCC_EXECUTABLE}")
    ENDIF()

    set(COMPILER_PATH ${CMAKE_CXX_COMPILER};${NVCC_COMPILER})
    list(REMOVE_DUPLICATES inc_dirs)
    list(REMOVE_DUPLICATES flags)
    list(REMOVE_DUPLICATES DEFS)
    list(REMOVE_DUPLICATES lib_dirs)
    list(REMOVE_DUPLICATES ${LIB_FILES_DEBUG_VAR})
    list(REMOVE_DUPLICATES ${LIB_FILES_RELEASE_VAR})

    set(inc "")
    foreach(dir "${inc_dirs}")
        set(inc "${dir}\n")
    endforeach(dir)
    set(lib "")
    foreach(dir "${lib_dirs}")
        set(lib "${dir}\n")
    endforeach(dir)
    if(WIN32)
        list(APPEND flags "/FS")
        list(REMOVE_ITEM lib ${CMAKE_BINARY_DIR})
    endif(WIN32)
    string(REGEX REPLACE ";" "\n" inc "${inc}")
    string(REGEX REPLACE "BUILD_INTERFACE:" "\n" inc "${inc}")
    string(REGEX REPLACE "\$<" "" inc "${inc}")
    string(REGEX REPLACE ">" "" inc "${inc}")
    string(REGEX REPLACE ";" "\n" lib "${lib}")
    string(REGEX REPLACE ";" "\n" flags "${flags}")
    string(REGEX REPLACE ";" "\n" defs "${DEFS}")
    string(REGEX REPLACE ";" "\n" deps "${deps}")

    if(MSVC)
        if(dest_dir_deb)
            FILE(WRITE "${dest_dir_deb}/${target}_config.txt"
                "project_id:\n-1\n\n"
                "include_dirs:\n${inc}\n"
                "lib_dirs_debug:\n${CMAKE_BINARY_DIR}/Debug\n${lib}\n"
                "lib_dirs_release:\n${lib}\n"
                "compile_options:\n${flags}\n\n"
                "compile_definitions:\n${DEFS}\n\n"
                "module_dependencies:\n${deps}\n\n"
                "compiler_location:\n${COMPILER_PATH}"
            )
        endif()
        if(dest_dir_rel)
            FILE(WRITE "${dest_dir_rel}/${target}_config.txt"
                "project_id:\n-1\n\n"
                "include_dirs:\n${inc}\n"
                "lib_dirs_debug:\n${lib}\n"
                "lib_dirs_release:\n${CMAKE_BINARY_DIR}/Release\n${lib}\n"
                "compile_options:\n${flags}\n\n"
                "compile_definitions:\n${DEFS}\n\n"
                "module_dependencies:\n${deps}\n\n"
                "compiler_location:\n${COMPILER_PATH}"
            )
        endif()
        if(dest_dir_reldeb)
            FILE(WRITE "${dest_dir_reldeb}/${target}_config.txt"
                "project_id:\n-1\n\n"
                "include_dirs:\n${inc}\n"
                "lib_dirs_debug:\n${lib}\n"
                "lib_dirs_release:\n${CMAKE_BINARY_DIR}/RelWithDebInfo\n${lib}\n"
                "compile_options:\n${flags}\n\n"
                "compile_definitions:\n${DEFS}\n\n"
                "module_dependencies:\n${deps}\n\n"
                "compiler_location:\n${COMPILER_PATH}"
            )
        endif()
            if(dest_dir)
                FILE(WRITE "${dest_dir}/${target}_config.txt"
                "project_id:\n-1\n\n"
                "include_dirs:\n${inc}\n"
                "lib_dirs_debug:\n${lib}\n"
                "lib_dirs_release:\n${lib}\n"
                "compile_options:\n${flags}\n\n"
                "compile_definitions:\n${DEFS}\n\n"
                "module_dependencies:\n${deps}\n\n"
                "compiler_location:\n${COMPILER_PATH}"
                )
            endif()
    else(MSVC)
        get_target_property(cxx_standard_reqd ${target} CXX_STANDARD_REQUIRED)
        if(cxx_standard_reqd)
            get_target_property(cxx_standard ${target} CXX_STANDARD)
            set(flags "${flags} -std=c++${cxx_standard}")
        endif()

        FILE(WRITE "${dest_dir}/${target}_config.txt"
            "project_id:\n-1\n\n"
            "include_dirs:\n${inc}\n"
            "lib_dirs_debug:\n${lib}\n"
            "lib_dirs_release:\n${lib}\n"
            "compile_options:\n${flags}\n\n"
            "compile_definitions:\n${DEFS}\n\n"
            "module_dependencies:\n${deps}\n\n"
            "compiler_location:\n${COMPILER_PATH}"
        )
    endif(MSVC)
endmacro(RCC_TARGET_CONFIG)

macro(WRITE_RCC_CONFIG RCC_INCLUDE_DEPENDENCIES RCC_LIBRARY_DIRS_DEBUG RCC_LIBRARY_DIRS_RELEASE RCC_COMPILE_FLAGS)

    string(REGEX REPLACE ";" "\n" includes_ "${RCC_INCLUDE_DEPENDENCIES}")
    #string(REGEX REPLACE "$<BUILD_INTERFACE:" "" includes_ "${includes_}")
    #string(REGEX REPLACE "" "" includes_ "${includes_}")
    string(REGEX REPLACE ";" "\n" lib_dirs_deb_ "${RCC_LIBRARY_DIRS_DEBUG}")
    string(REGEX REPLACE ";" "\n" lib_dirs_rel_ "${RCC_LIBRARY_DIRS_RELEASE}")
    if(RCC_COMPILE_FLAGS)
        string(REGEX REPLACE ";" "\n" flags_ "${RCC_COMPILE_FLAGS}")
    else()
        set(flags_ "")
    endif()
    set(NVCC_COMPILER "")
    IF(CUDA_FOUND)
        set(NVCC_COMPILER "${CUDA_NVCC_EXECUTABLE}")
    ENDIF()
    set(COMPILER_PATH ${CMAKE_CXX_COMPILER};${NVCC_COMPILER})
    IF(WIN32)
        FILE(WRITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/RCC_config.txt"
            "project_id:\n0\n"
            "include_dirs:\n${includes_}\n"
            "lib_dirs_debug:\n${lib_dirs_deb_}\n"
            "compile_options:\n/FS ${flags_}\n"
            "compiler_location:\n${COMPILER_PATH}")

        FILE(WRITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release/RCC_config.txt"
            "project_id:\n0"
            "\ninclude_dirs:\n${includes_}"
            "\nlib_dirs_release:\n${lib_dirs_rel_}"
            "\ncompile_options:\n/FS ${flags_}"
            "\ncompiler_location:\n${COMPILER_PATH}")

        FILE(WRITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/RelWithDebInfo/RCC_config.txt"
            "project_id:\n0"
            "\ninclude_dirs:\n${includes_}"
            "lib_dirs_release:\n${lib_dirs_rel_}"
            "compile_options:\n/FS ${flags_}"
            "compiler_location:\n${COMPILER_PATH}")

        FILE(WRITE "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/RCC_config.txt"
            "project_id:\n0"
            "\ninclude_dirs:\n${includes_}"
            "\nlib_dirs_debug:\n${lib_dirs_deb_}"
            "\nlib_dirs_release:\n${lib_dirs_rel_}"
            "\ncompile_options:\n\n/FS ${flags_}"
            "\ncompiler_location:\n${COMPILER_PATH}")
    ELSE(WIN32)
        FILE(WRITE "${CMAKE_BINARY_DIR}/RCC_config.txt"
            "project_id:\n0\n"
            "\ninclude_dirs:\n${includes_}\n"
            "\nlib_dirs_debug:\n${lib_dirs_deb_}\n${CMAKE_BINARY_DIR}\n"
            "\nlib_dirs_release:\n${lib_dirs_rel_}\n${CMAKE_BINARY_DIR}"
            "\ncompile_options:\n${flags_}\n"
            "\ncompiler_location:\n${COMPILER_PATH}")

        FILE(WRITE "${CMAKE_CURRENT_BINARY_DIR}/RCC_config.txt"
            "project_id:\n0\n"
            "\ninclude_dirs:\n${includes_}\n"
            "\nlib_dirs_debug:\n${lib_dirs_deb_}\n${CMAKE_BINARY_DIR}\n"
            "\nlib_dirs_release:\n${lib_dirs_rel_}\n${CMAKE_BINARY_DIR}"
            "\ncompile_options:\n${flags_}\n"
            "\ncompiler_location:\n${COMPILER_PATH}")
    ENDIF(WIN32)
endmacro()
