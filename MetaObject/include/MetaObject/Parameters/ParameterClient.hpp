#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <vector>
#include <string>

namespace mo
{
    class IVariableManager;
    class MO_EXPORTS ParameterClient
    {
    public:
        static ParameterClient* Instance();
        bool Connect(const std::string& parameter_server_address);
        
        // Subscribe a single variable known to this manager to the parameter server
        bool Subscribe(IVariableManager* mgr, const std::string& internal_name, const std::string& topic_name);

        // Remove a single variable known to this manager
        bool UnSubscribe(IVariableManager* mgr, const std::string& internal_name, const std::string& topic_name);

        std::vector<std::string> ListAvailableParameters(const std::string& name_filter = "");
    private:
        ParameterClient();
        ~ParameterClient();
        ParameterClient(const ParameterClient& other) = delete;
        ParameterClient& operator=(const ParameterClient& other) = delete;
        struct impl;
        impl* _pimpl;
    };
}