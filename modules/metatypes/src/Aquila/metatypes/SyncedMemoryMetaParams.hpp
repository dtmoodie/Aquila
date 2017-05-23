#pragma once
#include "Aquila/core/detail/Export.hpp"
#include "Aquila/types/SyncedMemory.hpp"
#include <MetaObject/params/TInputParam.hpp>

namespace mo
{
template<>
class AQUILA_EXPORTS TInputParamPtr<aq::SyncedMemory> : public ITInputParam<aq::SyncedMemory>
{
public:
    TInputParamPtr(const std::string& name = "", const aq::SyncedMemory** userVar_ = nullptr, Context* ctx = nullptr);
    bool setInput(std::shared_ptr<IParam> input);
    bool setInput(IParam* input);
    void setUserDataPtr(const aq::SyncedMemory** user_var_);
    bool getInput(mo::OptionalTime_t ts, size_t* fn = nullptr);
    bool getInput(size_t fn, mo::OptionalTime_t* ts = nullptr);
protected:
    const aq::SyncedMemory** userVar; // Pointer to the user space pointer variable of type T
    void updateUserVar();
    virtual void onInputUpdate(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, ICoordinateSystem*, UpdateFlags);
    virtual void onInputDelete(IParam const* param);
    aq::SyncedMemory current;
};
}
