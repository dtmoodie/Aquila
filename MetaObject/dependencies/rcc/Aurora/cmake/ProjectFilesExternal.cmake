
#
# RuntimeCompiler Source
#
MACRO(ADD_RCC_PROJECT name)



ENDMACRO()



aux_source_directory(Aurora/RuntimeCompiler RuntimeCompiler_SRCS)
aux_source_directory(Aurora/RuntimeCompiler/SimpleFileWatcher SimpleFileWatcher_SRCS)
FILE(GLOB_RECURSE compiler_headers "Aurora/RuntimeCompiler/*.h")

if(UNIX)
	list(REMOVE_ITEM RuntimeCompiler_SRCS "Aurora/RuntimeCompiler/Compiler_PlatformWindows.cpp")
	list(REMOVE_ITEM SimpleFileWatcher_SRCS "Aurora/RuntimeCompiler/SimpleFileWatcher/FileWatcherWin32.cpp")
	if(APPLE)
		list(REMOVE_ITEM SimpleFileWatcher_SRCS "Aurora/RuntimeCompiler/SimpleFileWatcher/FileWatcherLinux.cpp")
	else()
		list(REMOVE_ITEM SimpleFileWatcher_SRCS "Aurora/RuntimeCompiler/SimpleFileWatcher/FileWatcherOSX.cpp")
	endif()
else()
	list(REMOVE_ITEM RuntimeCompiler_SRCS "Aurora/RuntimeCompiler/Compiler_PlatformPosix.cpp")
	list(REMOVE_ITEM SimpleFileWatcher_SRCS "Aurora/RuntimeCompiler/SimpleFileWatcher/FileWatcherOSX.cpp")
	list(REMOVE_ITEM SimpleFileWatcher_SRCS "Aurora/RuntimeCompiler/SimpleFileWatcher/FileWatcherLinux.cpp")
endif()

set(RuntimeCompiler_SRCS ${RuntimeCompiler_SRCS} ${SimpleFileWatcher_SRCS} ${compiler_headers})

#
# RuntimeObjectSystem Source
#

aux_source_directory(Aurora/RuntimeObjectSystem RuntimeObjectSystem_SRCS)
aux_source_directory(Aurora/RuntimeObjectSystem/ObjectFactorySystem ObjectFactorySystem_SRCS)
aux_source_directory(Aurora/RuntimeObjectSystem/SimpleSerializer SimpleSerializer_SRCS)
FILE(GLOB_RECURSE object_system_headers "Aurora/RuntimeObjectSystem/*.h")

set(RuntimeCompiler_SRCS ${RuntimeCompiler_SRCS} ${ObjectFactorySystem_SRCS} ${SimpleSerializer_SRCS} ${object_system_headers})

if(UNIX)
	list(REMOVE_ITEM RuntimeObjectSystem_SRCS "Aurora/RuntimeObjectSystem/RuntimeObjectSystem_PlatformWindows.cpp")
else()
	list(REMOVE_ITEM RuntimeObjectSystem_SRCS "Aurora/RuntimeObjectSystem/RuntimeObjectSystem_PlatformPosix.cpp")
endif()

