#include <MetaObject/python/MetaObject.hpp>
#include <MetaObject/python/PythonConversionVisitation.hpp>

#include "nodes.hpp"
#include <Aquila/core/IGraph.hpp>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/nodes/Node.hpp>
#include <Aquila/nodes/NodeInfo.hpp>

#include <MetaObject/python/DataConverter.hpp>

#include <MetaObject/python/PythonSetup.hpp>
#include <MetaObject/python/rcc_ptr.hpp>

#include <ct/VariadicTypedef.hpp>

#include <RuntimeObjectSystem/IRuntimeObjectSystem.h>

#include <boost/functional.hpp>
#include <boost/python.hpp>
#include <boost/python/return_internal_reference.hpp>
#include <boost/python/signature.hpp>

namespace aq
{
    namespace python
    {
        template <int N, class Type, class S, class... Args>
        struct CreateNode
        {
        };
    } // namespace python
} // namespace aq

namespace boost
{
    namespace python
    {
        namespace detail
        {
            template <int N, class... T>
            struct is_keywords<aq::python::CreateNode<N, T...>>
            {
                static const bool value = true;
            };
        } // namespace detail
    }     // namespace python
} // namespace boost

namespace mo
{
    template <class... Args>
    struct StaticSlotAccessor<aq::nodes::Node::Ptr(Args...)>
    {
        using R = aq::nodes::Node::Ptr;
        template <class T, class BP>
        static void add(BP& bpobj, const IMetaObjectInfo* minfo)
        {
            auto static_slots = minfo->getStaticSlots();
            for (const auto& slot : static_slots)
            {
                if (slot.first->getSignature() == TypeInfo::create<R(Args...)>())
                {
                    auto tslot = dynamic_cast<TSlot<R(Args...)>*>(slot.first);
                    bpobj.def(slot.second.c_str(),
                              std::function<aq::python::GraphOwnerWrapper<aq::nodes::Node>(const Args&...)>(
                                  [tslot](const Args&... args) -> aq::python::GraphOwnerWrapper<aq::nodes::Node> {
                                      return {(*tslot)(args...)};
                                  }));

                    bpobj.staticmethod(slot.second.c_str());
                }
            }
        }
    };
} // namespace mo

namespace aq
{
    namespace python
    {

        static rcc::shared_ptr<aq::nodes::INode> getNode(const boost::python::object& bpobj)
        {
            rcc::shared_ptr<aq::nodes::INode> node;
            boost::python::extract<rcc::shared_ptr<aq::nodes::Node>> node_extract(bpobj);
            if (node_extract.check())
            {
                node = node_extract();
            }
            else
            {
                boost::python::extract<rcc::shared_ptr<aq::nodes::IFrameGrabber>> fg_extract(bpobj);
                if (fg_extract.check())
                {
                    node = fg_extract();
                }
            }
            return node;
        }

        template <class T>
        void connectInput(T& obj,
                          mo::ISubscriber* input,
                          const boost::python::object& bpobj,
                          const boost::python::object& queue_size)
        {
            auto input_param = dynamic_cast<mo::ISubscriber*>(input);
            if (!input_param)
            {
                return;
            }
            std::string name = obj.obj->getName();

            rcc::shared_ptr<aq::nodes::INode> node;
            mo::IPublisher* output_param = nullptr;
            node = getNode(bpobj);
            if (node)
            {
                if (node)
                {
                    output_param = node->getOutput("output");

                    if (!output_param || !input_param->acceptsPublisher(*output_param))
                    {
                        std::string input_name = input->getName();
                        output_param = node->getOutput(input_name);
                        if (!output_param || !input_param->acceptsPublisher(*output_param))
                        {
                            auto all_outputs = node->getOutputs();
                            uint32_t count = 0;
                            for (auto output : all_outputs)
                            {
                                if (input_param->acceptsPublisher(*output))
                                {
                                    output_param = output;
                                    ++count;
                                }
                            }
                            if (count > 1)
                            {
                                output_param = nullptr;
                            }
                        }
                    }
                }
            }
            else
            {
                if (boost::python::len(bpobj) > 1)
                {
                    node = getNode(bpobj[0]);
                    if (node)
                    {
                        boost::python::extract<mo::IPublisher*> param_extract(bpobj[1]);
                        if (param_extract.check())
                        {
                            output_param = param_extract();
                        }
                    }
                }
            }

            if (node && output_param)
            {
                if (!obj.obj->connectInput(input_param, node.get(), output_param))
                {
                    MO_LOG(debug,
                           "Unable to connect output {} to input {}",
                           output_param->getTreeName(),
                           input_param->getTreeName());
                }
                mo::IPublisher* pub = input_param->getPublisher();
                if (mo::buffer::IBuffer* buffer = dynamic_cast<mo::buffer::IBuffer*>(pub))
                {
                    // buffer->setFrameBufferCapacity()
                    boost::python::extract<float> float_queue_size(queue_size);
                    const bool is_float = Py_TYPE(queue_size.ptr()) == &PyFloat_Type;
                    if (is_float)
                    {

                        const float queue_size = float_queue_size();
                        mo::Duration queue_time = mo::ms * (queue_size * 1000.0F);
                        buffer->setTimePaddingCapacity(queue_time);
                    }
                    else
                    {
                        boost::python::extract<uint64_t> long_queue_size(queue_size);
                        if (long_queue_size.check())
                        {
                            const uint64_t queue_size = long_queue_size();
                            buffer->setFrameBufferCapacity(queue_size);
                        }
                    }
                }
                auto graph = node->getGraph();
                if (graph && !obj.graph)
                {
                    obj.graph = graph;
                }
            }
            else
            {
                if (!node)
                {
                    MO_LOG(debug, "Unable to find node");
                }
                if (!output_param)
                {
                    MO_LOG(debug, "Unable to find output for input {}", input->getName());
                }
            }
        }

        template <class T>
        void connectNamedInput(T& obj,
                               const std::string& name,
                               const boost::python::object& bpobj,
                               const boost::python::object& queue_size)
        {
            auto input = obj.obj->getInput(name);
            if (input)
            {
                connectInput(obj, input, bpobj, queue_size);
            }
        }

        template <class ConstructedType>
        static void initializeParametersAndInputs(ConstructedType& obj,
                                                  const std::vector<std::string>& param_names,
                                                  const std::vector<boost::python::object>& args,
                                                  IObjectConstructor* ctr,
                                                  const boost::python::object& queue_size)
        {
            MO_ASSERT(param_names.size() == args.size());
            for (size_t i = 0; i < param_names.size(); ++i)
            {
                if (args[i])
                {
                    mo::IControlParam* param = obj.obj->getParam(param_names[i]);
                    if (param)
                    {
                        mo::python::ControlParamSetter setter(args[i]);
                        param->load(setter);
                    }
                    else
                    {
                        mo::ISubscriber* input = obj.obj->getInput(param_names[i]);
                        if (input)
                        {
                            connectInput(obj, input, args[i], queue_size);
                            continue;
                        }
                    }
                }
            }
        }

        template <int N, class T, class Storage, class... Args>
        struct CreateNode<N, T, Storage, ct::VariadicTypedef<Args...>>
        {
            static const int size = N + 3;
            using ConstructedType = Storage;

            CreateNode(const std::vector<std::string>& param_names_)
            {
                MO_ASSERT_EQ(param_names_.size(), N);
                m_keywords[0] = (boost::python::arg("name") = "");
                m_keywords[1] = (boost::python::arg("graph") = boost::python::object());
                m_keywords[2] = (boost::python::arg("queue_size") = 0.1);
                for (size_t i = 0; i < param_names_.size(); ++i)
                {
                    m_keywords[i + 3] = (boost::python::arg(param_names_[i].c_str()) = boost::python::object());
                }
            }

            static ConstructedType create(IObjectConstructor* ctr,
                                          std::vector<std::string> param_names,
                                          const std::string& name,
                                          const boost::python::object& graph,
                                          const boost::python::object& queue_size,
                                          Args... args)
            {
                rcc::shared_ptr<T> ptr = ctr->Construct();
                ptr->Init(true);
                ConstructedType output;
                if (graph)
                {
                    boost::python::extract<rcc::shared_ptr<aq::IGraph>> graph_ext(graph);
                    if (graph_ext.check())
                    {
                        auto graph_ptr = graph_ext();
                        if (graph_ptr)
                        {
                            graph_ptr->addNode(ptr);
                            output.graph = graph_ptr;
                        }
                    }
                    else
                    {
                        boost::python::extract<GraphOwnerWrapper<aq::nodes::Node>> node_ext(graph);
                        if (node_ext.check())
                        {
                            auto graph_ptr = node_ext();
                            graph_ptr.obj->addChild(ptr);
                            output.graph = graph_ptr.graph;
                        }
                    }
                }
                output.obj = ptr;
                initializeParametersAndInputs(output, param_names, {args...}, ctr, queue_size);
                if (!name.empty())
                {
                    ptr->setName(name);
                }

                return output;
            }

            static std::function<ConstructedType(
                const std::string&, const boost::python::object&, const boost::python::object&, Args...)>
            bind(IObjectConstructor* ctr, std::vector<std::string> param_names)
            {
                return ctrBind(&CreateNode<N, T, Storage, ct::VariadicTypedef<Args...>>::create,
                               ctr,
                               param_names,
                               ct::make_int_sequence<size>{});
            }

            boost::python::detail::keyword_range range() const
            {
                return std::make_pair<boost::python::detail::keyword const*, boost::python::detail::keyword const*>(
                    &m_keywords[0], &m_keywords[0] + size);
            }

            std::array<boost::python::detail::keyword, size> m_keywords;
        };

        template <int N, class T, class Storage>
        struct CreateNode<N, T, Storage, ct::VariadicTypedef<void>>
        {
            static const int size = N + 3;
            using ConstructedType = Storage;

            CreateNode(const std::vector<std::string>& param_names_)
            {
                MO_ASSERT_EQ(param_names_.size(), N);
                m_keywords[0] = (boost::python::arg("name") = "");
                m_keywords[1] = (boost::python::arg("graph") = boost::python::object());
                m_keywords[2] = (boost::python::arg("queue_size") = 0.1);
                for (size_t i = 0; i < param_names_.size(); ++i)
                {
                    m_keywords[i + 3] = (boost::python::arg(param_names_[i].c_str()) = boost::python::object());
                }
            }

            static ConstructedType create(IObjectConstructor* ctr,
                                          std::vector<std::string> param_names,
                                          const std::string& name,
                                          const boost::python::object& graph,
                                          const boost::python::object& queue_size)
            {
                rcc::shared_ptr<T> ptr = ctr->Construct();
                if (ptr == nullptr)
                {
                    return ptr;
                }

                ptr->Init(true);
                ConstructedType output;
                if (graph)
                {
                    boost::python::extract<rcc::shared_ptr<aq::IGraph>> graph_ext(graph);
                    if (graph_ext.check())
                    {
                        auto graph_ptr = graph_ext();
                        if (graph_ptr)
                        {
                            graph_ptr->addNode(ptr);
                            output.graph = graph_ptr;
                        }
                    }
                    else
                    {
                        boost::python::extract<GraphOwnerWrapper<aq::nodes::Node>> node_ext(graph);
                        if (node_ext.check())
                        {
                            auto graph_ptr = node_ext();
                            graph_ptr.obj->addChild(ptr);
                            output.graph = graph_ptr.graph;
                        }
                    }
                }
                output.obj = ptr;
                initializeParametersAndInputs(output, param_names, {}, ctr, queue_size);
                if (!name.empty())
                {
                    ptr->setName(name);
                }

                return output;
            }

            static std::function<
                ConstructedType(const std::string&, const boost::python::object&, const boost::python::object&)>
            bind(IObjectConstructor* ctr, std::vector<std::string> param_names)
            {
                return [ctr, param_names](const std::string& name,
                                          const boost::python::object& graph,
                                          const boost::python::object& queue_size) {
                    return create(ctr, param_names, name, graph, queue_size);
                };
            }

            boost::python::detail::keyword_range range() const
            {
                return std::make_pair<boost::python::detail::keyword const*, boost::python::detail::keyword const*>(
                    &m_keywords[0], &m_keywords[0] + size);
            }

            std::array<boost::python::detail::keyword, size> m_keywords;
        };

        std::string printNode(const aq::nodes::INode* node)
        {
            auto params = node->getParams();
            std::stringstream ss;
            ss << node->getName();
            ss << '\n';
            for (auto param : params)
            {
                param->print(ss);
                ss << '\n';
            }
            return ss.str();
        }

        void setupNodeInterface()
        {
            static bool setup = false;
            if (setup)
            {
                return;
            }
            MO_LOG(info, "Registering INode to python");
            using INode = aq::nodes::INode;
            boost::python::
                class_<INode, rcc::shared_ptr<INode>, boost::python::bases<aq::IAlgorithm>, boost::noncopyable>
                    bpobj("INode", boost::python::no_init);
            bpobj.def("addChild", &INode::addChild);
            bpobj.def("getChildren", &INode::getChildren);
            /*bpobj.def(
                "connectInput",
                static_cast<bool (INode::*)(const std::string&, IMetaObject*, const std::string&, mo::BufferFlags)>(
                    &INode::connectInput),
                (boost::python::arg("output_node"),
                 boost::python::arg("output_name"),
                 boost::python::arg("input_name"),
                 boost::python::arg("connection_type") = mo::STREAM_BUFFER));*/
            bpobj.def("connectInput", &connectNamedInput<GraphOwnerWrapper<aq::nodes::Node>>);
            bpobj.def("__repr__", &printNode);
            setup = true;
        }

        struct NodeParamPolicy
        {
            std::vector<std::string> operator()(const std::vector<mo::ParamInfo*>& param_info)
            {
                std::vector<std::string> param_names;
                for (mo::ParamInfo* pinfo : param_info)
                {
                    if (!pinfo->getParamFlags().test(mo::ParamFlags::kOUTPUT))
                    {
                        param_names.push_back(pinfo->getName());
                    }
                }
                if (param_names.size() > 12)
                {
                    param_names.clear();
                    for (mo::ParamInfo* pinfo : param_info)
                    {
                        if (pinfo->getParamFlags().test(mo::ParamFlags::kINPUT))
                        {
                            param_names.push_back(pinfo->getName());
                        }
                    }
                }
                return param_names;
            }
        };

        std::vector<std::string> listInterfaceChildren(InterfaceID iid)
        {
            std::vector<std::string> out;
            auto ctrs = mo::MetaObjectFactory::instance()->getConstructors(iid);
            for (auto ctr : ctrs)
            {
                out.push_back(ctr->GetName());
            }
            return out;
        }

        void setupNodeObjects(std::vector<IObjectConstructor*>& ctrs)
        {
            using Node = aq::nodes::Node;

            boost::python::object module(
                boost::python::handle<>(boost::python::borrowed(PyImport_AddModule("aquila.nodes"))));

            boost::python::import("aquila").attr("nodes") = module;
            boost::python::scope plugins_scope = module;

            // auto system = mo::MetaObjectFactory::instance()->getObjectSystem();
            // auto ifaces = system->GetInterfaces();

            for (auto itr = ctrs.begin(); itr != ctrs.end();)
            {
                auto info = dynamic_cast<const aq::nodes::NodeInfo*>((*itr)->GetObjectInfo());
                if (info)
                {
                    const auto name = info->getDisplayName();
                    MO_LOG(debug, "Registering {} to python", name);
                    auto docstring = info->Print();
                    boost::python::class_<Node,
                                          GraphOwnerWrapper<Node>,
                                          boost::python::bases<aq::nodes::INode>,
                                          boost::noncopyable>
                        bpobj(name.c_str(), docstring.c_str(), boost::python::no_init);
                    auto ctr = mo::makeConstructor<Node, GraphOwnerWrapper<Node>, CreateNode>(*itr, NodeParamPolicy());
                    if (ctr)
                    {
                        bpobj.def("__init__", ctr);
                    }
                    else
                    {
                        bpobj.def("__init__",
                                  boost::python::make_constructor(std::function<GraphOwnerWrapper<Node>()>(
                                      std::bind(&constructWrappedObject<Node>, *itr))));
                    }
                    mo::addParamAccessors<Node>(bpobj, info);
                    mo::addOutputAccessors<Node>(bpobj, info);
                    mo::addSlotAccessors<Node, void>(bpobj, info);
                    mo::StaticSlotAccessor<std::vector<std::string>()>::add<Node>(bpobj, info);
                    mo::StaticSlotAccessor<aq::nodes::Node::Ptr(std::string)>::add<Node>(bpobj, info);
                    boost::python::import("aquila").attr("nodes").attr(info->GetObjectName().c_str()) = bpobj;

                    // Something like the following could be used to automatically add methods to list children of an
                    // interface
                    /*for(const auto& iface : ifaces)
                    {
                        if(name == iface.name)
                        {
                            bpobj.def("list",
                    std::function<std::vector<std::string>()>(std::bind(&listInterfaceChildren, iface.iid)));
                            bpobj.staticmethod("list");
                        }
                    }*/

                    itr = ctrs.erase(itr);
                }
                else
                {
                    ++itr;
                }
            }
            boost::python::implicitly_convertible<GraphOwnerWrapper<Node>, rcc::shared_ptr<Node>>();
        }
    } // namespace python
} // namespace aq
