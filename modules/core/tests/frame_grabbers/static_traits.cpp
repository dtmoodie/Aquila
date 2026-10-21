#include <Aquila/framegrabbers/GrabberInfo.hpp>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>

#include <MetaObject/core/SystemTable.hpp>

int32_t main(int32_t argc, char** argv)
{
    static_assert(grabbers::HasTimeout<aq::nodes::FrameGrabber>::value, "asdf");
    // loadTimeout reaches into the MetaObjectFactory singleton, which needs the
    // per-module system table wired up before any factory access
    auto table = SystemTable::instance();
    PerModuleInterface::GetInstance()->SetSystemTable(table.get());
    grabbers::hasTimeoutHelper<aq::nodes::FrameGrabber>();
}
