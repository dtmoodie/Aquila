#include "framegrabber.hpp"

#include <Aquila/core/Graph.hpp>
#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/python/MetaObject.hpp>

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include <boost/python/errors.hpp>
namespace aq
{
    namespace python
    {

        template <int N, class T, class Storage, class... Args>
        struct CreateFrameGrabber : public mo::CreateMetaObject<N + 2, T, Storage, Args...>
        {

            typedef Storage ConstructedType;

            CreateFrameGrabber(const std::vector<std::string>& param_names_)
            {
                MO_ASSERT_EQ(param_names_.size(), N);
                mo::CreateMetaObject<N + 2, T, Storage, Args...>::m_keywords[0] = (boost::python::arg("name") = "");
                mo::CreateMetaObject<N + 2, T, Storage, Args...>::m_keywords[1] =
                    (boost::python::arg("graph") = boost::python::object());
                for (size_t i = 0; i < param_names_.size(); ++i)
                {
                    mo::CreateMetaObject<N + 2, T, Storage, Args...>::m_keywords[i + 2] =
                        (boost::python::arg(param_names_[i].c_str()) = boost::python::object());
                }
            }

            static ConstructedType create(IObjectConstructor* ctr,
                                          std::vector<std::string> param_names,
                                          const std::string& name,
                                          const boost::python::object& graph,
                                          Args... args)
            {
                auto obj = mo::CreateMetaObject<N + 2, T, Storage, Args...>::create(ctr, param_names, args...);
                if (!name.empty())
                {
                    obj.obj->setTreeName(name);
                }

                if (graph)
                {
                    boost::python::extract<rcc::shared_ptr<aq::IGraph>> graph_ext(graph);
                    if (graph_ext.check())
                    {
                        auto graph_ptr = graph_ext();
                        if (graph_ptr)
                        {
                            graph_ptr->addNode(obj.obj);
                            obj.graph = graph_ptr;
                        }
                    }
                }
                return obj;
            }

            static std::function<ConstructedType(const std::string&, const boost::python::object&, Args...)>
            bind(IObjectConstructor* ctr, std::vector<std::string> param_names)
            {
                return ctrBind(&CreateFrameGrabber<N, T, Storage, Args...>::create,
                               ctr,
                               param_names,
                               ct::make_int_sequence<N + 2>{});
            }
        };

        void setupFrameGrabberInterface()
        {
            MO_LOG(info, "Registering INode to python");
        }

        std::vector<std::string> listDataSources()
        {
            auto out = aq::nodes::IFrameGrabber::listAllLoadableDocuments();
            std::vector<std::string> ret;
            for (auto& path_loader : out)
            {
                ret.push_back(path_loader.first);
            }
            return ret;
        }

        GraphOwnerWrapper<aq::nodes::IFrameGrabber> createFrameGrabber(const std::string& path,
                                                                       const std::string& preference = "")
        {
            return aq::nodes::IFrameGrabber::create(path, preference);
        }

        void setupFrameGrabberObjects(std::vector<IObjectConstructor*>& ctrs)
        {
            using FG = aq::nodes::IFrameGrabber;
            boost::python::object module(
                boost::python::handle<>(boost::python::borrowed(PyImport_AddModule("aquila.framegrabbers"))));

            boost::python::import("aquila").attr("framegrabbers") = module;
            boost::python::scope plugins_scope = module;

            for (auto itr = ctrs.begin(); itr != ctrs.end();)
            {
                auto info = dynamic_cast<const aq::nodes::FrameGrabberInfo*>((*itr)->GetObjectInfo());
                if (info)
                {
                    try
                    {
                        auto name = info->getDisplayName();
                        MO_LOG(debug, "Registering {} to python", name);
                        boost::python::class_<FG,
                                              GraphOwnerWrapper<FG>,
                                              boost::python::bases<aq::nodes::INode>,
                                              boost::noncopyable>
                            bpobj(name.c_str(), boost::python::no_init);
                        bpobj.def("__init__", mo::makeConstructor<FG, GraphOwnerWrapper<FG>>(*itr));
                        bpobj.def("loadData", static_cast<bool (FG::*)(std::string)>(&FG::loadData));
                        bpobj.def("loadData", static_cast<bool (FG::*)(std::vector<std::string>)>(&FG::loadData));
                        mo::addParamAccessors<FG>(bpobj, info);
                        boost::python::import("aquila").attr("framegrabbers").attr(info->GetObjectName().c_str()) =
                            bpobj;
                        itr = ctrs.erase(itr);
                    }
                    catch (const boost::python::error_already_set& e)
                    {
                        PyErr_Print();
                    }
                    catch (const std::exception& e)
                    {
                        std::cout << e.what() << std::endl;
                        itr = ctrs.erase(itr);
                    }
                }
                else
                {
                    ++itr;
                }
            }
            boost::python::implicitly_convertible<rcc::shared_ptr<FG>, rcc::shared_ptr<aq::nodes::INode>>();
            boost::python::implicitly_convertible<GraphOwnerWrapper<FG>, rcc::shared_ptr<FG>>();
            boost::python::def("listDataSources", &listDataSources);
            boost::python::def(
                "create", &createFrameGrabber, (boost::python::arg("path"), boost::python::arg("preference") = ""));
        }
    } // namespace python
} // namespace aq
