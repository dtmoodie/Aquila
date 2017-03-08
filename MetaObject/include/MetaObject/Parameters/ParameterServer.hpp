#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <string>


namespace mo
{
    class IVariableManager;
    class Context;
    class IParameter;
    class MO_EXPORTS ParameterServer
    {
    public:
        static ParameterServer* Instance();

        void SetTopic(const std::string& topic_name);
        
        // Publish all variables known to this manager
        bool Publish(IVariableManager* mgr);
        // Publish a single variable known to this manager
        bool Publish(IVariableManager* mgr, const std::string& parameter_name);

        // Remove all variables known to this manager
        bool Remove(IVariableManager* mgr);
        // Remove a single variable known to this manager
        bool Remove(IVariableManager* mgr, const std::string& parameter_name);

        // Bind the server to an adapter at a port
        bool Bind(const std::string& adapter);
    private:
        ParameterServer();
        ~ParameterServer();
        ParameterServer(const ParameterServer& other) = delete;
        ParameterServer& operator=(const ParameterServer& other) = delete;
        void onParameterUpdate(Context* ctx, IParameter* param);
        void onParameterDelete(IParameter const* param);
        struct impl;
        impl* _pimpl;
    };
}