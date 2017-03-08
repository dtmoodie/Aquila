#pragma once
#include "InputParameter.hpp"

namespace mo
{
class MO_EXPORTS InputParameterAny: public mo::InputParameter
{
public:
    InputParameterAny(const std::string& name = "");
    virtual bool GetInput(long long ts = -1);

    // This gets a pointer to the variable that feeds into this input
    virtual IParameter* GetInputParam();
    virtual bool SetInput(std::shared_ptr<mo::IParameter> param);
    virtual bool SetInput(mo::IParameter* param = nullptr);

    virtual bool AcceptsInput(std::weak_ptr<mo::IParameter> param) const;
    virtual bool AcceptsInput(mo::IParameter* param) const;
    virtual bool AcceptsType(mo::TypeInfo type) const;

    const mo::TypeInfo& GetTypeInfo() const;
    void on_param_update(mo::Context* ctx, mo::IParameter* param);
    void on_param_delete(mo::IParameter const *);
protected:
    IParameter* input = nullptr;
    static mo::TypeInfo _void_type_info;
    mo::TypedSlot<void(mo::Context*, mo::IParameter*)> _update_slot;
    mo::TypedSlot<void(mo::IParameter const*)> _delete_slot;
};

}
