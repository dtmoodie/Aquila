#pragma once
#ifdef HAVE_BOOST_PYTHON
#include "MetaObject/Detail/Export.hpp"
#include "shared_ptr.hpp"
#include <boost/python.hpp>

#include <functional>


namespace mo
{
    template<class T, int N, typename Enable = void> struct MetaObjectPolicy;
	class MO_EXPORTS PythonClassRegistry
	{
	public:
		static void SetupPythonModule();
		static void RegisterPythonSetupFunction(const char* name, std::function<void(void)> f);
	};

    template<class T> struct PythonPolicy
    {
        PythonPolicy(const char* name)
        {
            PythonClassRegistry::RegisterPythonSetupFunction(name, std::bind(&PythonPolicy<T>::PythonSetup, name));
        }
        static void PythonSetup(const char* name)
        {
            boost::python::class_<T, rcc::shared_ptr<T>, boost::noncopyable>(name, boost::python::no_init);
            boost::python::register_ptr_to_python<rcc::shared_ptr<T>>();
        }
    };

#define INSTANTIATE_PYTHON_POLICY_(N) \
template<class T> struct MetaObjectPolicy<T, N, void>: public mo::MetaObjectPolicy<T, N - 1, void>, public mo::PythonPolicy<T> \
{ \
    MetaObjectPolicy(): \
        PythonPolicy<T>(T::GetTypeNameStatic()) \
    { \
    } \
};

INSTANTIATE_PYTHON_POLICY_(__COUNTER__)
}
#endif