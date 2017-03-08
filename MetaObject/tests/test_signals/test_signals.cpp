#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Signals/RelayManager.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Parameters//ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"

#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"


#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif
#include <boost/thread.hpp>
#include <iostream>

using namespace mo;

BOOST_AUTO_TEST_CASE(signals)
{
	TypedSignal<int(int)> signal;
	{
		TypedSlot<int(int)> slot([](int val)
		{
			return val * 2;
		});
		signal.Connect(&slot);

		BOOST_CHECK_EQUAL(signal(4), 8);
	}
	BOOST_CHECK_THROW(signal(4), std::string);	
}

BOOST_AUTO_TEST_CASE(threaded_signal)
{
    mo::Context ctx;
    mo::Context thread_ctx;

    TypedSlot<void(int)> slot = TypedSlot<void(int)>(std::bind(
        [&thread_ctx](int value)->void
        {
            BOOST_REQUIRE_EQUAL(thread_ctx.thread_id, mo::GetThisThread());
            BOOST_REQUIRE_EQUAL(5, value);
        }, std::placeholders::_1));

    slot.SetContext(&thread_ctx);

    TypedSignal<void(int)> signal;
    auto connection = slot.Connect(&signal);

    boost::thread thread = boost::thread([&thread_ctx]()->void
    {
        thread_ctx.thread_id = mo::GetThisThread();
        while(!boost::this_thread::interruption_requested())
        {
            ThreadSpecificQueue::Run(thread_ctx.thread_id);
        }
    });

    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    signal(&ctx, 5);
    thread.interrupt();
    thread.join();
}

BOOST_AUTO_TEST_CASE(relay_manager)
{
    mo::Context ctx;
    mo::RelayManager manager;

    
}