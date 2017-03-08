file(GLOB qt_gui_src "src/*.cpp")
file(GLOB qt_gui_hdr "include/*.h" "include/*.hpp")
FILE(GLOB qt_gui_forms "ui/*.ui")
QT5_WRAP_UI(UIS ${qt_gui_forms})
QT5_WRAP_CPP(qt_gui_MOC ${qt_gui_hdr})
include_directories("include")
include_directories(${CMAKE_CURRENT_BINARY_DIR})


add_executable(example_qt_gui ${qt_gui_MOC} ${UIS} ${qt_gui_src} ${qt_gui_hdr})

set_target_properties(example_qt_gui PROPERTIES FOLDER examples)
target_link_libraries(example_qt_gui MetaObject Qt5::Core Qt5::Widgets Qt5::Gui instantiations)
if(OpenCV_FOUND)
  target_link_libraries(example_qt_gui ${OpenCV_LIBS})

endif()
if(MSVC)
      CONFIGURE_FILE("../../tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/examples/qt_gui/example_qt_gui.vcxproj.user @ONLY)
endif()