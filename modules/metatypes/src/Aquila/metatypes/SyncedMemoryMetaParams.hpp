#pragma once
#include "Aquila/Detail/Export.hpp"
#include "Aquila/types/SyncedMemory.hpp"
#include <MetaObject/params/TInputParam.hpp>

namespace mo
{
template<>
class AQUILA_EXPORTS TypedInputParameterPtr<aq::SyncedMemory> : public ITypedInputParameter<aq::SyncedMemory>
{
public:
    TypedInputParameterPtr(const std::string& name = "", const aq::SyncedMemory** userVar_ = nullptr, Context* ctx = nullptr);
    bool SetInput(std::shared_ptr<IParameter> input);
    bool SetInput(IParameter* input);
    void SetUserDataPtr(const aq::SyncedMemory** user_var_);
    bool GetInput(boost::optional<mo::time_t> ts, size_t* fn = nullptr);
    bool GetInput(size_t fn, boost::optional<mo::time_t>* ts = nullptr);
protected:
    const aq::SyncedMemory** userVar; // Pointer to the user space pointer variable of type T
    void updateUserVar();
    virtual void onInputUpdate(Context* ctx, IParameter* param);
    virtual void onInputDelete(IParameter const* param);
    aq::SyncedMemory current;
};
}
