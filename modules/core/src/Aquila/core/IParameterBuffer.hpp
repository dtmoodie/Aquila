#pragma once
#include "MetaObject/detail/TypeInfo.hpp"
#include <MetaObject/logging/logging.hpp>
#include <boost/any.hpp>

namespace aq
{
    class IParameterBuffer
    {
    public:
        virtual void setBufferSize(int size) = 0;
        virtual boost::any& getParameter(mo::TypeInfo, const std::string& name, int frameNumber) = 0;

        template<typename T> bool getParameter(T& param, const std::string& name, int frameNumber)
        {
            auto& parameter = getParameter(mo::TypeInfo(typeid(T)), name, frameNumber);
            if (parameter.empty())
                return false;
            try
            {
                param = boost::any_cast<T>(parameter);
                return true;
            }
            catch (boost::bad_any_cast& bad_cast)
            {
                MO_LOG(trace) << bad_cast.what();
            }
            return false;
        }
        template<typename T> bool setParam(T& param, const std::string& name, int frameNumber)
        {
            auto& parameter = getParameter(mo::TypeInfo(typeid(T)), name, frameNumber);
            parameter = param;
            return true;
        }
    };

}
