#include "MetaObject/Parameters/ParameterServer.hpp"
#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"
#include "MetaObject/IO/ZeroMQ.hpp"
#include "MetaObject/IO/ZeroMQImpl.hpp"
#include "MetaObject/Signals/TypedSlot.hpp"
#include "MetaObject/Parameters/Demangle.hpp"
#include "MetaObject/Parameters/IVariableManager.h"
#include "MetaObject/Logging/Log.hpp"
#include <cereal/archives/binary.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <boost/thread.hpp>
#include <map>

using namespace mo;

struct ParameterServer::impl
{
#ifdef HAVE_ZEROMQ
    impl(zmq::context_t& ctx):
        publisher(ctx, ZMQ_XPUB)
    {
        topic = "ParameterTopic";
        publisher.bind("tcp://*:5566");
        receive_thread = boost::thread(boost::bind(&ParameterServer::impl::messageLoop, this));
    }

    ~impl()
    {
        receive_thread.interrupt();
        receive_thread.join();
    }
    // This only handls subscribing and unsubscribing from parameter updates
    void messageLoop()
    {
        zmq::message_t message;
        static const char sub[] = "SUB:";
        static const char unsub[] = "UNSUB:";
        static const char list[] = "LIST";
        static const char topics[] = "TOPICS";
        publisher.setsockopt(ZMQ_RCVTIMEO, 1000);
        while(!boost::this_thread::interruption_requested())
        {
            if(publisher.recv(&message))
            {
                const char* data = static_cast<const char*>(message.data());
                if(memcmp(data, sub, sizeof(sub)) == 0)
                {
                    
                    // Get the rest of the topic
                    const char* topic_name = data + sizeof(sub);
                    // Find the parameter
                    for(auto& itr : sub_info)
                    {
                        if(itr.first->GetTreeName().compare(0, message.size() - sizeof(sub), topic_name) == 0)
                        {
                            ++itr.second.subscription_count;
                            itr.first->Subscribe();
                            LOG(debug) << "Subscriber added to " << itr.first->GetTreeName();
                            break;
                        }
                    }
                }else if(memcmp(data, unsub, sizeof(unsub)) == 0)
                {
                    const char* topic_name = data + sizeof(unsub);
                    for(auto& itr : sub_info)
                    {
                        if(itr.first->GetName().compare(0, message.size() - sizeof(sub), topic_name) == 0)
                        {
                            --itr.second.subscription_count;
                            itr.first->Unsubscribe();
                            LOG(debug) << "Subscriber removed from " << itr.first->GetTreeName();
                            break;
                        }
                    }
                }else if(memcmp(data, list, sizeof(list)) == 0)
                {
                    LOG(debug) << "Negotiating connection";
                    std::stringstream ss;
                    Demangle::GetTypeMapBinary(ss);
                    cereal::BinaryOutputArchive ar(ss);
                    
                    std::vector<std::pair<std::string, std::string>> data;
                    data.reserve(published_parameters.size());
                    for(auto& itr : published_parameters)
                    {
                        data.emplace_back(Demangle::TypeToName(itr->GetTypeInfo()), itr->GetName());
                    }
                    ar(data);
                    std::string msg = ss.str();
                    zmq::message_t msg_(msg.data(), msg.size());
                    zmq::message_t topic(topics, sizeof(topics));
                    publisher.send(topic, ZMQ_SNDMORE);
                    publisher.send(msg_);
                }
            }
        }
    }
    boost::thread receive_thread;
    zmq::socket_t publisher;

    TypedSlot<void(Context*, IParameter*)> update_slot;
    TypedSlot<void(IParameter const*)> delete_slot;

    struct subscription_info
    {
        int subscription_count = 0;
    };
    std::map<IParameter*, subscription_info> sub_info;
    
    std::list<IParameter*> published_parameters;
#endif
    std::string topic;
};

ParameterServer::ParameterServer()
{
    auto inst = ZeroMQContext::Instance();
    if(inst)
    {
#ifdef HAVE_ZEROMQ
        _pimpl = new impl(inst->_pimpl->ctx);
        _pimpl->update_slot = std::bind(&ParameterServer::onParameterUpdate, this, std::placeholders::_1, std::placeholders::_2);
        _pimpl->delete_slot= std::bind(&ParameterServer::onParameterDelete, this, std::placeholders::_1);
#else
        _pimpl = nullptr;
#endif
    }
    
    
}

ParameterServer::~ParameterServer()
{
    delete _pimpl;
}

ParameterServer* ParameterServer::Instance()
{
    static ParameterServer g_inst;
    return &g_inst;
}

void ParameterServer::SetTopic(const std::string& topic_name)
{
    _pimpl->topic = topic_name;
}


bool ParameterServer::Publish(IVariableManager* mgr)
{
    return false;
}


bool ParameterServer::Publish(IVariableManager* mgr, const std::string& parameter_name)
{
#ifdef HAVE_ZEROMQ
    auto param = mgr->GetParameter(parameter_name);
    if(param)
    {
        _pimpl->published_parameters.push_back(param);
        // TODO handle callback of param update here

        return true;
    }
#endif
    return false;
}


bool ParameterServer::Remove(IVariableManager* mgr)
{
    return false;
}


bool ParameterServer::Remove(IVariableManager* mgr, const std::string& parameter_name)
{
    return false;
}


bool ParameterServer::Bind(const std::string& adapter)
{
    return false;
}

void ParameterServer::onParameterUpdate(Context* ctx, IParameter* param)
{

}

void ParameterServer::onParameterDelete(IParameter const* param)
{

}