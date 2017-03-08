#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Parameters/InputParameter.hpp"

namespace mo
{
    class ParameterServer;
    class ParameterClient;
    class MO_EXPORTS ZeroMQContext
    {
    public:
        static ZeroMQContext* Instance();
    protected:
        friend class ParameterServer;
        friend class ParameterClient;
        struct impl;
        impl* _pimpl;
    private:
        ZeroMQContext();
        ZeroMQContext(const ZeroMQContext& ctx) = delete;
        ZeroMQContext& operator=(const ZeroMQContext& ctx) = delete;
    };


    class MO_EXPORTS ParameterPublisher: public InputParameter
    {
    public:
        ParameterPublisher();
        virtual ~ParameterPublisher();

        virtual bool GetInput(long long ts = -1) = 0;

        // This gets a pointer to the variable that feeds into this input
        virtual IParameter* GetInputParam() = 0;

        // Set input and setup callbacks
        virtual bool SetInput(std::shared_ptr<IParameter> param) = 0;
        virtual bool SetInput(IParameter* param = nullptr) = 0;

        // Check for correct serialization routines, etc
        virtual bool AcceptsInput(std::weak_ptr<IParameter> param) const = 0;
        virtual bool AcceptsInput(IParameter* param) const = 0;
        virtual bool AcceptsType(TypeInfo type) const = 0;
    protected:
        void onInputUpdate(Context* ctx, IParameter* param);
        struct impl;
        impl* _pimpl;
    };   
}