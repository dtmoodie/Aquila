#include "graph.hpp"
#include "nodes.hpp"

#include <Aquila/types/SyncedMemory.hpp>

#include <Aquila/core.hpp>
#include <Aquila/core/IAlgorithm.hpp>
#include <Aquila/core/IGraph.hpp>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/nodes/INode.hpp>
#include <Aquila/nodes/NodeInfo.hpp>

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

namespace aq
{
    namespace python
    {

        void setupGraphInterface()
        {
            static bool setup = false;
            if (setup)
            {
                return;
            }
            MO_LOG(info, "Registering IGraph to python");
            setup = true;
        }

        template <class T>
        void addNode(const rcc::shared_ptr<aq::IGraph>& graph, GraphOwnerWrapper<T>& wrapper)
        {
            graph->addNode(wrapper.obj);
            wrapper.graph = graph;
        }

        std::vector<std::string> getSlots(const rcc::shared_ptr<aq::IGraph>& graph)
        {
            std::vector<std::string> out;
            auto mgr = graph->getRelayManager();
            if (mgr)
            {
                auto relays = mgr->getAllRelays();
                for (const auto& pair : relays)
                {
                    std::stringstream ss;
                    ss << pair.second << " - " << mo::TypeTable::instance()->typeToName(pair.first->getSignature());
                    out.push_back(ss.str());
                }
            }
            return out;
        }

        void emitsignal(const rcc::shared_ptr<aq::IGraph>& graph, const std::string& name)
        {
            auto mgr = graph->getRelayManager();
            auto relay = mgr->getRelayOptional<void(void)>(name);
            if (relay)
            {
                auto current = graph->getStream();
                (*relay)(current.get());
            }
            else
            {
                auto all_relays = mgr->getAllRelays();
                std::string suggestions;
                for (const auto& relay : all_relays)
                {
                    if (relay.second.find(name) != std::string::npos)
                    {
                        suggestions += relay.second + ", ";
                    }
                }
                if (!suggestions.empty())
                {
                    MO_LOG(info, "Unable to find relay with name {} did you mean: {}", name, suggestions);
                }
            }
        }

        rcc::shared_ptr<aq::IGraph> makeGraph(IObjectConstructor* ctr)
        {
            rcc::shared_ptr<aq::IGraph> output;
            auto obj = ctr->Construct();
            if (obj)
            {
                output = obj;
                output->Init(true);
                auto table = SystemTable::instance();
                if (table)
                {
                    auto graphs = table->getSingleton<std::vector<rcc::weak_ptr<aq::IGraph>>>();
                    if (!graphs)
                    {
                        graphs = std::make_shared<std::vector<rcc::weak_ptr<aq::IGraph>>>();
                        table->setSingleton(graphs);
                    }
                    if (graphs)
                    {
                        graphs->push_back(output);
                    }
                }
            }
            return output;
        }

        bool waitSignal(const rcc::shared_ptr<aq::IGraph>& graph, const std::string& name, int timeout)
        {
            bool received = false;
            if (graph)
            {
                auto mgr = graph->getRelayManager();

                if (mgr->getRelayOptional<void(void)>(name))
                {
                    auto relay = mgr->getRelay<void(void)>(name);
                    boost::condition_variable cv;
                    boost::mutex mtx;
                    mo::TSlot<void(void)> slot([&cv, &received]() {
                        received = true;
                        cv.notify_all();
                    });
                    slot.connect(relay);
                    if (timeout == 0)
                    {
                        boost::unique_lock<boost::mutex> lock(mtx);
                        cv.wait(lock);
                    }
                    else
                    {
                        boost::unique_lock<boost::mutex> lock(mtx);
                        cv.wait_for(lock, boost::chrono::milliseconds(timeout));
                    }
                }
                else
                {
                    MO_LOG(info, "{} not a valid signal", name);
                }
            }
            return received;
        }

        void setupGraphObjects(std::vector<IObjectConstructor*>& ctrs)
        {
            for (auto itr = ctrs.begin(); itr != ctrs.end();)
            {
                if ((*itr)->GetObjectInfo()->GetObjectName() == "Graph")
                {
                    boost::python::object module(
                        boost::python::handle<>(boost::python::borrowed(PyImport_AddModule("aquila"))));

                    boost::python::class_<aq::IGraph,
                                          rcc::shared_ptr<aq::IGraph>,
                                          boost::python::bases<mo::IMetaObject>,
                                          boost::noncopyable>
                        bpobj("Graph", boost::python::no_init);
                    bpobj.def("__init__",
                              boost::python::make_constructor(
                                  std::function<rcc::shared_ptr<aq::IGraph>()>(std::bind(&makeGraph, *itr))));
                    bpobj.def("save",
                              boost::python::make_function(&aq::IGraph::saveGraph,
                                                           boost::python::default_call_policies(),
                                                           boost::python::arg("filename")));
                    bpobj.def("start", &aq::IGraph::start);
                    bpobj.def("stop", &aq::IGraph::stop);

                    bpobj.def("step", &aq::IGraph::process);
                    // bpobj.def("addNode", static_cast<void(aq::IGraph::*) (const
                    // rcc::shared_ptr<aq::nodes::INode>&)>(&aq::IGraph::addNode));
                    bpobj.def("addNode", addNode<aq::nodes::Node>);
                    bpobj.def("addNode", addNode<aq::nodes::IFrameGrabber>);
                    bpobj.def("listSlots", &getSlots);
                    bpobj.def("emit", &emitsignal);
                    bpobj.def("wait", &waitSignal, (boost::python::arg("name"), boost::python::arg("timeout") = 0));
                    bpobj.add_property("dirty", &aq::IGraph::getDirty);

                    boost::python::import("aquila").attr("Graph") = bpobj;
                    ctrs.erase(itr);
                    return;
                }
                ++itr;
            }
        }
    } // namespace python
} // namespace aq
