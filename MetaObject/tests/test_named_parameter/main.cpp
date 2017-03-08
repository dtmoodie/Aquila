
#define BOOST_TEST_MAIN

#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Detail/IMetaObjectImpl.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Parameters//ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/Parameters/Types.hpp"
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE "parameter"
#include <boost/test/included/unit_test.hpp>
#endif

#include <iostream>

using namespace mo;
struct PrintConstruct
{
    PrintConstruct()
    {
        std::cout << "construct\n";
    }
    int member = 0;
};
#define MO_KEYWORD_INPUT(name, type) \
namespace tag \
{ \
    struct name \
    { \
        typedef type Type; \
        typedef const Type& ConstRef; \
        typedef Type& Ref; \
        typedef ConstRef StorageType; \
        typedef const void* VoidType; \
    }; \
    static mo::kwargs::TypedKeyword<name>& _##name = \
            mo::kwargs::TypedKeyword<name>::instance; \
}

#define MO_KEYWORD_OUTPUT(name, type) \
namespace tag \
{ \
    struct name \
    { \
        typedef type Type; \
        typedef const Type& ConstRef; \
        typedef Type& Ref; \
        typedef Ref StorageType; \
        typedef void* VoidType; \
    }; \
    static mo::kwargs::TypedKeyword<name>& _##name = \
            mo::kwargs::TypedKeyword<name>::instance; \
}

namespace mo
{
    namespace kwargs
    {
        template<class Tag>
        struct TaggedArgument
        {
            typedef Tag TagType;
            explicit TaggedArgument(typename Tag::StorageType val):
                arg(val)
            {

            }
            typename Tag::VoidType get() const
            {
                return &arg;
            }
        protected:
            typename Tag::StorageType arg;
        };

        template<class Tag>
        struct TypedKeyword
        {
            static TypedKeyword instance;
            TaggedArgument<Tag> operator=(typename Tag::StorageType data)
            {
                return TaggedArgument<Tag>(data);
            }
        };
        template<class T> TypedKeyword<T> TypedKeyword<T>::instance;
    }
    MO_KEYWORD_INPUT(timestamp, double);
    MO_KEYWORD_INPUT(frame_number, size_t);
    MO_KEYWORD_INPUT(dummy, PrintConstruct);
    MO_KEYWORD_OUTPUT(output, PrintConstruct);
}



template<class Tag>
typename Tag::VoidType GetKeyImpl()
{
    return 0;
}



template<class Tag, class T, class... Args>
typename Tag::VoidType GetKeyImpl(const T& arg, const Args&... args)
{
    return std::is_same<typename T::TagType, Tag>::value ?
                const_cast<void*>(arg.get()) :
                const_cast<void*>(GetKeyImpl<Tag, Args...>(args...));
}

template<class Tag, class... Args>
typename Tag::ConstRef GetInput(const Args&... args)
{
    const void* ptr = GetKeyImpl<Tag>(args...);
    assert(ptr);
    return *(static_cast<const typename Tag::Type*>(ptr));
}

template<class Tag, class... Args>
typename Tag::ConstRef GetInputDefault(typename Tag::ConstRef def, const Args&... args)
{
    const void* ptr = GetKeyImpl<Tag>(args...);
    if(ptr)
        return *(const typename Tag::Type*)ptr;
    return def;
}

template<class Tag, class... Args>
const typename Tag::Type* GetInputOptional(const Args&... args)
{
    const void* ptr = GetKeyImpl<Tag>(args...);
    if(ptr)
        return (const typename Tag::Type*)ptr;
    return nullptr;
}

template<class Tag, class... Args>
typename Tag::Ref GetOutput(const Args&... args)
{
    static_assert(!std::is_const<typename Tag::VoidType>::value, "Tag type is not an output tag");
    void* ptr = GetKeyImpl<Tag>(args...);
    assert(ptr);
    return *(static_cast<typename Tag::Type*>(ptr));
}

template<class Tag, class... Args>
typename Tag::Type* GetOutputOptional(const Args&... args)
{
    static_assert(!std::is_const<typename Tag::VoidType>::value, "Tag type is not an output tag");
    void* ptr = GetKeyImpl<Tag>(args...);
    if(ptr)
        return (static_cast<typename Tag::Type*>(ptr));
    return nullptr;
}

template<class... Args>
void keywordFunction(const Args&... args)
{
    const size_t& fn = GetInput<tag::frame_number>(args...);
    const double& timestamp = GetInput<tag::timestamp>(args...);
    const PrintConstruct& pc = GetInput<tag::dummy>(args...);
    std::cout << "Frame number: " << fn << "\n";
    std::cout << "Timestamp: " << timestamp << std::endl;
    PrintConstruct& pc_out = GetOutput<tag::output>(args...);
    pc_out.member = fn;
}


BOOST_AUTO_TEST_CASE(named_parameter)
{
    size_t fn = 100;
    PrintConstruct pc;
    keywordFunction(tag::_frame_number = fn, tag::_timestamp = 0.5, tag::_dummy = pc, tag::_output = pc);
    fn = 200;
    keywordFunction(tag::_frame_number = fn, tag::_dummy = pc, tag::_timestamp = 1.0, tag::_output = pc);

}


