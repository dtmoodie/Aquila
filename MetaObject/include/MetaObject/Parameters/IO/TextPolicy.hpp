#pragma once
#include <MetaObject/Parameters/IParameter.hpp>
#include <MetaObject/Parameters/MetaParameter.hpp>
#include "SerializationFunctionRegistry.hpp"
#include <boost/lexical_cast.hpp>

namespace mo 
{ 
namespace IO 
{ 
namespace Text 
{
    namespace imp
    {
        // test if stream serialization of a type is possible
        template<class T>
        struct stream_serializable
        {
            //const static bool value = sizeof(decltype(std::declval<std::istream>() >> std::declval<T>(), size_t())) == sizeof(size_t);
            template<class U>
            static constexpr auto check(std::stringstream is, U val, int)->decltype(is >> val, size_t())
            {
                return 0;
            }
            template<class U>
            static constexpr int check(std::stringstream is, U val, size_t)
            {
                return 0;
            }
            static const bool value = sizeof(check<T>(std::stringstream(), std::declval<T>(), 0)) == sizeof(size_t);
        };


        template<typename T>
        auto Serialize_imp(std::ostream& os, T const& obj, int) ->decltype(os << obj, void())
        {
            os << obj;
        }

        template<typename T>
        void Serialize_imp(std::ostream& os, T const& obj, long)
        {

        }

        template<typename T>
        auto DeSerialize_imp(std::istream& is, T& obj, int) ->decltype(is >> obj, void())
        {
            is >> obj;
        }
        template<typename T>
        void DeSerialize_imp(std::istream& is, T& obj, long)
        {

        }

        template<typename T>
        auto Serialize_imp(std::ostream& os, std::vector<T> const& obj, int)->decltype(os << std::declval<T>(), void())
        {
            os << "[";
            for(int i = 0; i < obj.size(); ++i)
            {
                if(i != 0)
                    os << ", ";
                os << obj[i];
            }
            os << "]";
        }

        template<typename T>
        auto DeSerialize_imp(std::istream& is, std::vector<T>& obj, int) ->decltype(is >> std::declval<T>(), void())
        {
            std::string str;
            is >> str;
            auto pos = str.find('=');
            if(pos != std::string::npos)
            {
                size_t index = boost::lexical_cast<size_t>(str.substr(0, pos));
                T value = boost::lexical_cast<T>(str.substr(pos + 1));
                if(index < obj.size())
                {

                }else
                {
                    obj.resize(index + 1);
                    obj[index] = value;
                }
            }
        }

        template<class T1, class T2>
        typename std::enable_if<stream_serializable<T1>::value && stream_serializable<T2>::value >::type
        DeSerialize_imp(std::istream& is, std::map<T1, T2>& obj, int)
        {
            std::string str;
            is >> str;
            auto pos = str.find('=');
            if(pos == std::string::npos)
                return;
            std::stringstream ss;
            ss << str.substr(0, pos);
            T1 key;
            ss >> key;
            ss.str("");
            ss << str.substr(pos + 1);
            T2 value;
            ss >> value;
            obj[key] = value;
        }


        template<typename T>
        bool Serialize(ITypedParameter<T>* param, std::stringstream& ss)
        {
            T* ptr = param->GetDataPtr();
            if (ptr)
            {
                Serialize_imp(ss, *ptr, 0);
                //ss << *ptr;
                return true;
            }
            return false;
        }

        template<typename T>
        bool DeSerialize(ITypedParameter<T>* param, std::stringstream& ss)
        {
            T* ptr = param->GetDataPtr();
            if (ptr)
            {
                //ss >> *ptr;
                DeSerialize_imp(ss, *ptr, 0);
                return true;
            }
            return false;
        }
        template<typename T> bool Serialize(ITypedParameter<std::vector<T>>* param, std::stringstream& ss)
        {
            std::vector<T>* ptr = param->GetDataPtr();
            if (ptr)
            {
                ss << ptr->size();
                ss << "[";
                for (size_t i = 0; i < ptr->size(); ++i)
                {
                    if (i != 0)
                        ss << ", ";
                    ss << (*ptr)[i];
                }
                ss << "]";
                return true;
            }
            return false;
        }
        template<typename T> bool DeSerialize(ITypedParameter<std::vector<T>>* param, std::stringstream& ss)
        {
            std::vector<T>* ptr = param->GetDataPtr();
            if (ptr)
            {
                auto pos = ss.str().find('=');
                if(pos != std::string::npos)
                {
                    std::string str;
                    std::getline(ss, str, '=');
                    size_t index = boost::lexical_cast<size_t>(str);
                    std::getline(ss, str);
                    T value = boost::lexical_cast<T>(str);
                    if(index >= ptr->size())
                    {
                        ptr->resize(index + 1);
                    }
                    (*ptr)[index] = value;
                    return true;
                }else
                {
                    ptr->clear();
                    std::string size;
                    std::getline(ss, size, '[');
                    if (size.size())
                    {
                        ptr->reserve(boost::lexical_cast<size_t>(size));
                    }
                    T value;
                    char ch; // For flushing the ','
                    while (ss >> value)
                    {
                        ss >> ch;
                        ptr->push_back(value);
                    }
                }
                return true;
            }
            return false;
        }
    }
    

    template<typename T> bool WrapSerialize(IParameter* param, std::stringstream& ss)
    {
        ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
        if (typed)
        {
            if(imp::Serialize(typed, ss))
            {
                return true;
            }
        }
        return false;
    }

    template<typename T> bool WrapDeSerialize(IParameter* param, std::stringstream& ss)
    {
        ITypedParameter<T>* typed = dynamic_cast<ITypedParameter<T>*>(param);
        if (typed)
        {
            if(imp::DeSerialize(typed, ss))
            {
                typed->Commit();
                return true;
            }
        }
        return false;
    }
    
    template<class T> struct Policy
    {
        Policy()
        {
            SerializationFunctionRegistry::Instance()->SetTextSerializationFunctions(
                TypeInfo(typeid(T)),
                std::bind(&WrapSerialize<T>, std::placeholders::_1, std::placeholders::_2),
                std::bind(&WrapDeSerialize<T>, std::placeholders::_1, std::placeholders::_2));
        }
    };
} // Text 
} // IO

#define PARAMETER_TEXT_SERIALIZATION_POLICY_INST_(N) \
  template<class T> struct MetaParameter<T, N, void>: public MetaParameter<T, N - 1, void> \
    { \
        static IO::Text::Policy<T> _text_policy;  \
        MetaParameter(const char* name): \
            MetaParameter<T, N-1, void>(name) \
        { \
            (void)&_text_policy; \
        } \
    }; \
    template<class T> IO::Text::Policy<T> MetaParameter<T, N, void>::_text_policy;

PARAMETER_TEXT_SERIALIZATION_POLICY_INST_(__COUNTER__)
} // mo
