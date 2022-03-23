#include <Aquila/framegrabbers/GrabberInfo.hpp>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>

int32_t main(int32_t argc, char** argv)
{
    static_assert(grabbers::HasTimeout<aq::nodes::FrameGrabber>::value, "asdf");
    grabbers::hasTimeoutHelper<aq::nodes::FrameGrabber>();
}
