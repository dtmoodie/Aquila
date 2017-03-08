#include "MetaObject/Parameters/ParameterClient.hpp"
#include "MetaObject/IO/ZeroMQImpl.hpp"
#include "MetaObject/IO/StreamView.hpp"
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/utility.hpp>
#include <map>
using namespace mo;

struct ParameterClient::impl
{
    
};

ParameterClient::ParameterClient()
{
    _pimpl = new impl();
}

ParameterClient::~ParameterClient()
{
    delete _pimpl;
}

ParameterClient* ParameterClient::Instance()
{
    static ParameterClient g_inst;
    return &g_inst;
}

bool ParameterClient::Connect(const std::string& parameter_server_address)
{
    auto inst = ZeroMQContext::Instance();
    if(inst)
    {
#ifdef HAVE_ZEROMQ
        zmq::socket_t socket(inst->_pimpl->ctx, ZMQ_SUB);
        static const char topics[] = "TOPICS";
        static const char list[] = "LIST";
        socket.connect(parameter_server_address);
        socket.setsockopt(ZMQ_SUBSCRIBE, topics, sizeof(topics));
        zmq::message_t msg(topics, sizeof(topics));
        socket.send(msg);
        socket.setsockopt(ZMQ_RCVTIMEO, 1000);
        zmq::message_t message;
        
        if(socket.recv(&message))
        {
            mo::StreamView view((char*)message.data(), message.size());
            std::map<std::string, size_t> type_map;
            std::vector<std::pair<std::string, std::string>> data;
            std::istream stream(&view);
            cereal::BinaryInputArchive ar(stream);
            ar(type_map);
            ar(data);
        }
#endif
    }
    return false;
}


bool ParameterClient::Subscribe(IVariableManager* mgr, const std::string& internal_name, const std::string& topic_name)
{
    return false;
}


bool ParameterClient::UnSubscribe(IVariableManager* mgr, const std::string& internal_name, const std::string& topic_name)
{
    return false;
}

std::vector<std::string> ParameterClient::ListAvailableParameters(const std::string& name_filter)
{
    return std::vector<std::string>();
}
    
