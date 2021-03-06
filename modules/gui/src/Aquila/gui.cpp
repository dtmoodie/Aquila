#include "gui.hpp"
#include <Aquila/gui/UiCallbackHandlers.h>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/object/MetaObjectFactory.cpp>

#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>

#include <opencv2/core.hpp>

namespace aq {
namespace gui {

void initModule(mo::MetaObjectFactory *factory) {
  factory->setupObjectConstructors(PerModuleInterface::GetInstance());
}

void guiThreadFunc()
{
    mo::ThreadRegistry::instance()->registerThread(mo::ThreadRegistry::GUI);
    MO_LOG(info, "GUI thread {}", mo::getThisThread());
    boost::mutex dummy_mtx; // needed for cv
    boost::condition_variable cv;
    auto notifier = mo::ThreadSpecificQueue::registerNotifier([&cv]() { cv.notify_all(); });
    while (!boost::this_thread::interruption_requested())
    {
        mo::setThreadName("Aquila GUI thread");
        try
        {
            boost::mutex::scoped_lock lock(dummy_mtx);
            cv.wait_for(lock, boost::chrono::milliseconds(30));
            mo::ThreadSpecificQueue::run();
        }
        catch (boost::thread_interrupted& err)
        {
            (void)err;
            break;
        }
        catch (mo::ExceptionWithCallStack<cv::Exception>& e)
        {
            MO_LOG(debug) << "Opencv exception with callstack " << e.what() << " " << e.callStack();
        }
        catch (mo::IExceptionWithCallStackBase& e)
        {
            MO_LOG(debug) << "Exception with callstack " << e.callStack();
        }
        catch (cv::Exception& e)
        {
            MO_LOG(debug) << "OpenCV exception: " << e.what();
        }
        catch (...)
        {
            MO_LOG(debug) << "Unknown / unhandled exception thrown in gui thread event handler";
        }
        try
        {
            aq::WindowCallbackHandler::EventLoop::Instance()->run();
        }
        catch (mo::ExceptionWithCallStack<cv::Exception>& e)
        {
            (void)e;
        }
        catch (cv::Exception& e)
        {
            (void)e;
        }
        catch (boost::thread_interrupted& /*err*/)
        {
            break;
        }
        catch (...)
        {
        }
    try
    {
      aq::WindowCallbackHandler::EventLoop::Instance()->run();
    } catch (cv::Exception &e) {
      (void)e;
    } catch (boost::thread_interrupted & /*err*/) {
      break;
    } catch (...) {
    }
  }
  mo::ThreadSpecificQueue::cleanup();
  MO_LOG(info, "Gui thread shutting down naturally");
}

boost::thread createGuiThread() { return boost::thread(guiThreadFunc); }
}
}
