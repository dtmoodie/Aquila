#pragma once
#include "Aquila/core/detail/Export.hpp"
#include "Aquila/types/SyncedMemory.hpp"
#include <MetaObject/params/TInputParam.hpp>

namespace mo
{
template<>
class AQUILA_EXPORTS TInputParamPtr<aq::SyncedMemory> : virtual public ITInputParam<aq::SyncedMemory> {
public:
    typedef typename ParamTraits<aq::SyncedMemory>::Storage_t Storage_t;
    typedef typename ParamTraits<aq::SyncedMemory>::ConstStorageRef_t ConstStorageRef_t;
    typedef typename ParamTraits<aq::SyncedMemory>::InputStorage_t InputStorage_t;
    typedef typename ParamTraits<aq::SyncedMemory>::Input_t Input_t;
    typedef void(TUpdateSig_t)(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags);
    typedef TSignal<TUpdateSig_t> TUpdateSignal_t;
    typedef TSlot<TUpdateSig_t> TUpdateSlot_t;

    TInputParamPtr(const std::string& name = "", Input_t* userVar_ = nullptr, Context* ctx = nullptr);
    bool setInput(std::shared_ptr<IParam> input);
    bool setInput(IParam* input);
    void setUserDataPtr(Input_t* user_var_);
    bool getInput(const OptionalTime_t& ts, size_t* fn = nullptr);
    bool getInput(size_t fn, OptionalTime_t* ts = nullptr);

protected:
    virtual bool updateDataImpl(const Storage_t&, const OptionalTime_t&, Context*, size_t, const std::shared_ptr<ICoordinateSystem>&) {
        return true;
    }
    Input_t* _user_var; // Pointer to the user space pointer variable of type T
    InputStorage_t _current_data;
    virtual void onInputUpdate(ConstStorageRef_t, IParam*, Context*, OptionalTime_t, size_t, const std::shared_ptr<ICoordinateSystem>&, UpdateFlags);
};
}
