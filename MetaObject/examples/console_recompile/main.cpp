#include "obj.hpp"

#include <MetaObject/MetaObjectFactory.hpp>
#include <MetaObject/Logging/Log.hpp>
#include <boost/thread.hpp>
int main()
{

    try
    {
        THROW(debug) << "Test throw";

    }catch(mo::ExceptionWithCallStack<std::string>& e)
    {
        LOG(debug) << "Exception caught in the correct handler";
    }catch(...)
    {
        LOG(debug) << "Exception caught in the wrong handler";
    }

	auto factory = mo::MetaObjectFactory::Instance(); // ->RegisterTranslationUnit();
	factory->RegisterTranslationUnit();
	auto obj = rcc::shared_ptr<printable>::Create();
	
	bool recompiling = false;
	while (1)
	{
		obj->print();


		if (factory->CheckCompile())
		{
			recompiling = true;
		}
		if (recompiling)
		{
			if (factory->SwapObjects())
			{
				recompiling = false;
			}
		}
		boost::this_thread::sleep_for(boost::chrono::seconds(1));
	}

}
