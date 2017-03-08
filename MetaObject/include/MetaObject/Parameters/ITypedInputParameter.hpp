#pragma once

#include "InputParameter.hpp"
#include "ITypedParameter.hpp"
#ifdef _MSC_VER
#pragma warning( disable : 4250)
#endif
namespace mo
{
    template<class T> class ITypedInputParameter: virtual public ITypedParameter<T>, virtual public InputParameter
    {
    public:
        ITypedInputParameter(const std::string& name = "",  Context* ctx = nullptr);
        ~ITypedInputParameter();
        bool SetInput(std::shared_ptr<IParameter> input);
        bool SetInput(IParameter* input);

        virtual bool AcceptsInput(std::weak_ptr<IParameter> param) const;
        virtual bool AcceptsInput(IParameter* param) const;
        virtual bool AcceptsType(TypeInfo type) const;
        IParameter* GetInputParam();
        
        bool GetInput(long long ts);

        T* GetDataPtr(long long ts = -1, Context* ctx = nullptr);
        bool GetData(T& value, long long time_step = -1, Context* ctx = nullptr);
        T GetData(long long ts = -1, Context* ctx = nullptr);


        ITypedParameter<T>* UpdateData(T& data_, long long ts, Context* ctx);
        ITypedParameter<T>* UpdateData(const T& data_, long long ts, Context* ctx);
        ITypedParameter<T>* UpdateData(T* data_, long long ts, Context* ctx);

    protected:
        virtual void onInputDelete(IParameter const* param);
        virtual void onInputUpdate(Context* ctx, IParameter* param);
        std::shared_ptr<ITypedParameter<T>> shared_input;
        ITypedParameter<T>* input;

    private:
		TypedSlot<void(Context*, IParameter*)> update_slot;
		TypedSlot<void(IParameter const*)> delete_slot;
    };
}
#include "detail/ITypedInputParameterImpl.hpp"