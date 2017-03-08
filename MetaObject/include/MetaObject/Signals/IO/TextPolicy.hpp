#pragma once
#include <MetaObject/Signals/TypedSlot.hpp>
#include <MetaObject/Signals/TypedSignalRelay.hpp>
#include <MetaObject/Signals/Serialization.hpp>
namespace mo
{
    template<int ...S, class R, class ... T>
    void call_signal(seq<S...>, std::tuple<T...>& params, mo::TypedSlot<R(T...)>* sig)
    {
        (*sig)(std::get<S>(params)...);
    }
    
    template<int ... S, class R, class ... T>
    void call_signal(seq<S...>, std::tuple<T...>& params, mo::TypedSignalRelay<R(T...)>* relay)
    {
        (*relay)(std::get<S>(params)...);
    }

    template<typename T> auto SerializeImpl(std::ostream& ss, T& val, unsigned int)->decltype(ss << val, void)
    {
    }
    template<typename T> auto SerializeImpl(std::ostream& ss, T& val, int)->decltype(ss << val, void)
    {
        ss << " ! " << val;
    }

    // ********************* deserialize SFINAE *****************************************
    template<typename T> void DeserializeImpl(std::istream& ss, T& val, unsigned int)
    {

    }
    template<typename T> auto DeserializeImpl(std::istream& ss, T& val, int) ->decltype(ss >> val, void)
    {
        std::string tmp;
        std::getline(ss, tmp, '!');
        std::stringstream tmpss;
        tmpss << tmp;
        tmpss >> val;
    }

    template<int N, class... T> class TupleSerializer
    {
    public:
        static void deserialize(std::ostream& ss, std::tuple<T...>& args)
        {
            TupleSerializer<N - 1, T...>::deserialize(ss, args);
            DeserializeImpl(ss, std::get<N>(args), 0);
        }
        static void serialize(std::istream& ss, std::tuple<T...>& args)
        {
            TupleSerializer<N - 1, T...>::serialize(ss, args);
            SerializeImpl(ss, std::get<N>(args), 0);
        }
    };

    template<class... T> class TupleSerializer<0, T...>
    {
    public:
        static void Deserialize(std::ostream& ss, std::tuple<T...>& args)
        {
            DeserializeImpl(ss, std::get<0>(args), 0);
        }
        static void Serialize(std::istream& ss, std::tuple<T...>& args)
        {
            SerializeImpl(ss, std::get<0>(args), 0);
        }
    };

    // Specialization for a signal accepting only one parameter
    template<> class MO_EXPORTS TupleSerializer<0, std::string>
    {
    public:
        static void Deserialize(std::ostream& ss, std::tuple<std::string>& args);
        static void Serialize(std::istream& ss, std::tuple<std::string>& args);
    };

    template<class T> class TextSlotCaller;

    template<class R, class ... T> class TextSlotCaller<R(T...)>: public ISignalCaller
    {
    public:
        static ISignalCaller* Create(ISlot* slot)
        {
            auto typed = dynamic_cast<TypedSlot<R(T...)>*>(slot);
            if(typed)
            {
                return new TextSlotCaller<R(T...)>(slot);
            }
        }

        TextSlotCaller(TypedSlot<R(T...)>* slot)
        {
            _slot = slot;
        }

        void Call(const std::istream& ss)
        {
            std::tuple<T...> params;
            TupleSerializer<T...>::Deserialize(ss, params);
        }

    private:
        TypedSlot<R(T...)>* _slot;
    };

    template<class T> class TextSlotSink;

    template<class R, class ... T> class TextSlotSink: virtual public ISignalSink, virtual public TypedSlot<R(T...)>
    {
    public:
        static ISignalSink* Create(std::shared_ptr<ISignalRelay> relay, std::ostream& stream)
        {
            auto typed = std::dynamic_pointer_cast<TypedSignalRelay<R(T...)>*>(relay);
            if(typed)
            {
                return new TextSlotSink<T>(typed, stream);
            }
        }

        TextSlotSink(std::shared_ptr<TypedSignalRelay<R(T...)>> relay, std::ostream& stream):
            _stream(stream),
            _relay(relay)
        {
            relay->Connect(this);
        }
        
        ~TextSlotSink()
        {
            
        }
    private:
        std::shared_ptr<TypedSignalRelay<R(T...)>> _relay;
        std::ostream& _stream;
    };
}