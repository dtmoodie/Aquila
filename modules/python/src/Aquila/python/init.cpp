#include "init.hpp"
#include "algorithm.hpp"
#include "framegrabber.hpp"
#include "graph.hpp"
#include "nodes.hpp"

#include <Aquila/types/SyncedMemory.hpp>

#include <Aquila/core.hpp>
#include <Aquila/core/IAlgorithm.hpp>
#include <Aquila/core/IGraph.hpp>
#include <Aquila/core/KeyValueStore.hpp>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/nodes/INode.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <Aquila/serialization.hpp>

#include <MetaObject/MetaParameters.hpp>
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/object/RelayManager.hpp>
#include <MetaObject/python/MetaObject.hpp>
#include <MetaObject/python/PythonSetup.hpp>
#include <MetaObject/python/rcc_ptr.hpp>
#include <MetaObject/thread/Thread.hpp>
#include <MetaObject/thread/ThreadRegistry.hpp>

#include <boost/functional.hpp>
#include <boost/python.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/signature.hpp>

#include <thread>

bool recompile(bool async = false)
{
    auto inst = mo::MetaObjectFactory::instance();
    if (inst->checkCompile())
    {
        MO_LOG(info, "Currently compiling");
        if (async == false)
        {
            while (!inst->isCompileComplete())
            {
                boost::this_thread::sleep_for(boost::chrono::seconds(1));
                MO_LOG(info, "Still compiling");
            }
            MO_LOG(info, "Swapping objects");
            auto graphs = SystemTable::instance()->getSingleton<std::vector<rcc::weak_ptr<aq::IGraph>>>();
            if (graphs)
            {
                for (auto& graph : *graphs)
                {
                    auto shared = graph.lock();
                    if (shared)
                    {
                        shared->stop();
                    }
                }
            }
            bool success = inst->swapObjects();
            if (success)
            {
                MO_LOG(info, "Swap success");
            }

            if (graphs)
            {
                for (auto& graph : *graphs)
                {
                    auto shared = graph.lock();
                    if (shared)
                    {
                        shared->start();
                    }
                }
            }
            return success;
        }
    }
    else
    {
        MO_LOG(info, "Nothing to recompile");
    }
    return false;
}

struct AqLibGuard
{
    AqLibGuard(std::shared_ptr<SystemTable> table)
        : m_table(std::move(table))
    {
        // gui_thread = aq::gui::createGuiThread();
    }

    ~AqLibGuard()
    {
        gui_thread.interrupt();
        gui_thread.join();
    }

    boost::thread gui_thread;
    std::shared_ptr<SystemTable> m_table;
};

void readArgs(const boost::python::list& args)
{
    std::vector<std::string> argv;
    for (ssize_t i = 0; i < boost::python::len(args); ++i)
    {
        argv.push_back(boost::python::extract<std::string>(args[i])());
    }
    aq::KeyValueStore::instance()->parseArgs(std::move(argv));
}

#define BOOST_PYTHON_USE_GCC_SYMBOL_VISIBILITY

BOOST_PYTHON_MODULE(aquila)
{
    auto table = mo::python::pythonSetup("aquila");
    boost::shared_ptr<AqLibGuard> lib_guard(new AqLibGuard(table));
    mo::python::setLogLevel("debug");
    mo::python::RegisterInterface<aq::IAlgorithm> alg(&aq::python::setupAlgorithmInterface,
                                                      &aq::python::setupAlgorithmObjects);
    mo::python::RegisterInterface<aq::nodes::INode> node(&aq::python::setupNodeInterface,
                                                         &aq::python::setupNodeObjects);
    mo::python::RegisterInterface<aq::nodes::IFrameGrabber> fg(&aq::python::setupFrameGrabberInterface,
                                                               &aq::python::setupFrameGrabberObjects);
    mo::python::RegisterInterface<aq::IGraph> graph(&aq::python::setupGraphInterface, &aq::python::setupGraphObjects);

    mo::initMetaParamsModule(table.get());
    auto factory = mo::MetaObjectFactory::instance(table.get());
    factory->registerTranslationUnit();
    aq::core::initModule(factory.get());
    // aq::gui::initModule(factory);
    aq::serialization::initModule(table.get());
    boost::python::def("readArgs", &readArgs);
    boost::python::def("recompile", &recompile, (boost::python::arg("async") = false));

    boost::python::class_<AqLibGuard, boost::shared_ptr<AqLibGuard>, boost::noncopyable>("aqLibGuard",
                                                                                         boost::python::no_init);

    boost::python::scope().attr("__aqlibguard") = lib_guard;
}
