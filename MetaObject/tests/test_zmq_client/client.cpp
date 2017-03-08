#ifdef HAVE_ZEROMQ


#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Detail/IMetaObjectImpl.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Parameters/ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/Parameters/TypedParameter.hpp"
#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"
#include "MetaObject/Parameters/ParameterClient.hpp"
#include "cereal/archives/portable_binary.hpp"
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"


#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif
#include <iostream>
#include "zmq.hpp"
#include "zmq_utils.h"
using namespace mo;


BOOST_AUTO_TEST_CASE(client)
{
    auto inst = ParameterClient::Instance();
    inst->Connect("tcp://localhost:5566");
    


    /*zmq::context_t ctx(1);

    zmq::socket_t socket(ctx, ZMQ_SUB);
    socket.connect("tcp://localhost:5566");
    const char* topic = "update_topic";
    socket.setsockopt(ZMQ_SUBSCRIBE, topic, strlen(topic));
    zmq::message_t msg;
    TypedParameter<int> parameter;
    parameter.UpdateData(0);
    auto deserialization_func = SerializationFunctionRegistry::Instance()->GetBinaryDeSerializationFunction(parameter.GetTypeInfo());
    int count = 0;
    while(1)
    {
        if(socket.recv(&msg))
        {
            std::istringstream iss(static_cast<char*>(msg.data()));
            cereal::BinaryInputArchive ar(iss);
            deserialization_func(&parameter, ar);
            long long ts = parameter.GetTimestamp();
            int value = parameter.GetData();
            BOOST_REQUIRE_EQUAL(ts, value);
            ++count;
            if(count > 100)
                break;
        }
    }*/
}





#else
#include <iostream>
int main()
{
    std::cout << "Not build with zero mq supprt";
    return 0;
}
#endif
