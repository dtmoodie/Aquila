if(WITH_WT)
find_package(Wt QUIET)
  if(Wt_FOUND)
    find_path(Wt_BIN_DIR
      wt.dll
      HINTS "${Wt_INCLUDE_DIR}/../bin"
    )
    set(MO_HAVE_WT 1 CACHE BOOL INTERNAL FORCE)

    if(Wt_BIN_DIR)
      set(Wt_BIN_DIR_DBG ${Wt_BIN_DIR} CACHE PATH "" FORCE)
      set(Wt_BIN_DIR_OPT ${Wt_BIN_DIR} CACHE PATH "" FORCE)
      set(bin_dirs_ "${BIN_DIRS};Wt")
      list(REMOVE_DUPLICATES bin_dirs_)
      set(BIN_DIRS "${bin_dirs_}" CACHE STRING "" FORCE)
    endif(Wt_BIN_DIR)
  endif(Wt_FOUND)
endif(WITH_WT)
