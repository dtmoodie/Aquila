#include <Aquila/Nodes/Node.h>
#include <Aquila/Nodes/IFrameGrabber.hpp>
#include <Aquila/Nodes/NodeFactory.h>
#include <Aquila/DataStream.hpp>

#include <boost/python.hpp>
#include <boost/python/raw_function.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

using namespace std;
using namespace aq;


// --------------------------------------------------------------
// Here are a bunch of static functions that can be called from python

std::vector<std::string> ListConstructableNodes(std::string filter)
{
    return aq::Nodes::Node::ListConstructableNodes(filter);
}
std::vector<std::string> ListConstructableNodes1()
{
    return aq::Nodes::Node::ListConstructableNodes("");
}
std::vector<std::string> ListDevices()
{
    return aq::Nodes::IFrameGrabber::ListAllLoadableDocuments();
}

std::vector<std::string> ListHistory()
{
    return std::vector<std::string>();
}

rcc::shared_ptr<IDataStream> open_datastream(string source)
{
    std::string doc = source;
    if(aq::IDataStream::CanLoadDocument(doc))
    {
        LOG(debug) << "Found a frame grabber which can load " << doc;
        auto stream = aq::IDataStream::Create(doc);
        if(stream->LoadDocument(doc))
        {
            stream->StartThread();
            return stream;
        }else
        {
            LOG(warning) << "Unable to load document";
        }
    }else
    {
        LOG(warning) << "Unable to find a frame grabber which can load " << doc;
    }
    return rcc::shared_ptr<IDataStream>();
}

rcc::shared_ptr<Nodes::Node> create_node(string name)
{
    return aq::NodeFactory::Instance()->AddNode(name);
}

namespace boost
{
    namespace python
    {
        template<typename T> struct pointee<rcc::shared_ptr<T>>
        {
            typedef T type;
        };
    }
}
namespace rcc
{
    template<typename T> T* get_pointer(rcc::shared_ptr<T> & p)
    {
        return p.Get();
    }
    template<typename T> const T* get_pointer(const rcc::shared_ptr<T> & p)
    {
        return p.Get();
    }
    template<typename T> T* get_pointer(rcc::weak_ptr<T> & p)
    {
        return p.Get();
    }
}


BOOST_PYTHON_MODULE(EaglePython)
{
    boost::python::scope().attr("__version__") = "0.1";

    boost::python::def("ListConstructableNodes", &ListConstructableNodes);
    boost::python::def("ListConstructableNodes", &ListConstructableNodes1);
    boost::python::def("ListDevices", &ListDevices);


    //boost::python::class_<EagleLib::DataStream, rcc::shared_ptr<EagleLib::DataStream>, boost::noncopyable>("DataStream", boost::python::no_init)
        //.def("__init__", boost::python::make_constructor(&open_datastream))
        //.def("GetName", &EagleLib::Nodes::Node::getName)
        //.def("GetFullName", &EagleLib::Nodes::Node::getFullTreeName);
        //.def("GetParameters", &EagleLib::Nodes::Node::getParameters);

    boost::python::class_<aq::Nodes::Node, rcc::shared_ptr<aq::Nodes::Node>, boost::noncopyable>("Node", boost::python::no_init)
        .def("__init__", boost::python::make_constructor(&create_node));


    //boost::python::class_<Parameters::Parameter, boost::noncopyable>("Parameter", boost::python::no_init)
      //  .def("GetName", &Parameters::Parameter::GetName);
        
    
    //boost::python::register_ptr_to_python<rcc::shared_ptr<EagleLib::DataStream>>();

    boost::python::register_ptr_to_python<rcc::shared_ptr<aq::Nodes::Node>>();

    boost::python::class_<vector<mo::IParameter*>>("ParamVec")
        .def(boost::python::vector_indexing_suite<vector<mo::IParameter*>, true>());

}
