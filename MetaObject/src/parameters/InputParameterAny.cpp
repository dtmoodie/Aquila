#include "MetaObject/Parameters/InputParameterAny.hpp"
using namespace mo;

InputParameterAny::InputParameterAny(const std::string& name):
    _update_slot(std::bind(&InputParameterAny::on_param_update, this, std::placeholders::_1, std::placeholders::_2)),
    _delete_slot(std::bind(&InputParameterAny::on_param_delete, this, std::placeholders::_1))
{
    this->SetName(name);
    _void_type_info = mo::TypeInfo(typeid(void));
    this->AppendFlags(mo::Input_e);
}

bool InputParameterAny::GetInput(long long ts)
{
    return true;
}


IParameter* InputParameterAny::GetInputParam()
{
    return input;
}

bool InputParameterAny::SetInput(std::shared_ptr<mo::IParameter> param)
{
    input = param.get();
    Commit();
    return true;
}
bool InputParameterAny::SetInput(mo::IParameter* param)
{
    input = param;
    param->RegisterDeleteNotifier(&_delete_slot);
    param->RegisterUpdateNotifier(&_update_slot);
    Commit();
    return true;
}

bool InputParameterAny::AcceptsInput(std::weak_ptr<mo::IParameter> param) const
{
    return true;
}

bool InputParameterAny::AcceptsInput(mo::IParameter* param) const
{
    return true;
}

bool InputParameterAny::AcceptsType(mo::TypeInfo type) const
{
    return true;
}

const mo::TypeInfo& InputParameterAny::GetTypeInfo() const
{
    if(input)
        return input->GetTypeInfo();
    return _void_type_info;
}

void InputParameterAny::on_param_update(mo::Context* ctx, mo::IParameter* param)
{
    Commit(-1, ctx); // Notify owning parent of update
}

void InputParameterAny::on_param_delete(mo::IParameter const *)
{
    input = nullptr;
}

TypeInfo InputParameterAny::_void_type_info;
