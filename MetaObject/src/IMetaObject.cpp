#include "MetaObject/IMetaObject.hpp"

#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Parameters/Demangle.hpp"
#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/Signals/ISlot.hpp"
#include "MetaObject/Signals/SignalInfo.hpp"
#include "MetaObject/Signals/SlotInfo.hpp"
#include "MetaObject/Signals/RelayManager.hpp"
#include "MetaObject/Detail/IMetaObject_pImpl.hpp"
#include "MetaObject/Detail/IMetaObjectImpl.hpp"
#include "MetaObject/Parameters/IParameter.hpp"
#include "MetaObject/Parameters/InputParameter.hpp"
#include "MetaObject/Parameters/Buffers/BufferFactory.hpp"
#include "MetaObject/Parameters/InputParameter.hpp"
#include "MetaObject/Parameters/VariableManager.h"

#include "ISimpleSerializer.h"
#include "IObjectState.hpp"

#include <boost/thread/recursive_mutex.hpp>

using namespace mo;
int IMetaObject::Connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name)
{
	int count = 0;
    auto my_signals = sender->GetSignals(signal_name);
    auto my_slots = receiver->GetSlots(slot_name);
	
    for (auto signal : my_signals)
	{
        for (auto slot : my_slots)
		{
			if (signal->GetSignature() == slot->GetSignature())
			{
				auto connection = slot->Connect(signal);
				if (connection)
				{
					sender->AddConnection(connection, signal_name, slot_name, slot->GetSignature(), receiver);
					++count;
				}
				break;
            }else
            {
                LOG(debug) << "Signature mismatch, Slot (" << slot_name << " -  " <<  slot->GetSignature().name()
                           << ") != Signal (" << signal_name << " - " << signal->GetSignature().name() << ")";
            }
		}
	}
	
	return count;
}

bool IMetaObject::Connect(IMetaObject* sender, const std::string& signal_name, IMetaObject* receiver, const std::string& slot_name, const TypeInfo& signature)
{
	auto signal = sender->GetSignal(signal_name, signature);
	if (signal)
	{
		auto slot = receiver->GetSlot(slot_name, signature);
		if (slot)
		{
			auto connection = slot->Connect(signal);
			sender->AddConnection(connection, signal_name, slot_name, signature, receiver);
			return true;
		}
	}
	return false;
}

IMetaObject::IMetaObject()
{
    _mtx = new boost::recursive_mutex();
    _pimpl = new impl();
    _ctx = nullptr;
    _sig_manager = nullptr;
    _pimpl->_slot_parameter_updated = std::bind(&IMetaObject::onParameterUpdate, this, std::placeholders::_1, std::placeholders::_2);
}


IMetaObject::~IMetaObject()
{
    delete _mtx;
    delete _pimpl;
}

void IMetaObject::Init(bool firstInit)
{
    InitParameters(firstInit);
	InitSignals(firstInit);
    BindSlots(firstInit);
    InitCustom(firstInit);
    auto params = GetParameters();
    for(auto param : params)
    {
        auto update_slot = this->GetSlot<void(mo::Context*, mo::IParameter*)>("on_" + param->GetName() + "_modified");
        if(update_slot)
        {
            auto connection = param->RegisterUpdateNotifier(update_slot);
            this->AddConnection(connection, param->GetName() + "_modified", "on_" + param->GetName() + "_modified", update_slot->GetSignature(), this);
        }
        auto delete_slot = this->GetSlot<void(mo::IParameter const*)>("on_" + param->GetName() + "_deleted");
        if(delete_slot)
        {
            auto connection = param->RegisterDeleteNotifier(delete_slot);
            this->AddConnection(connection, param->GetName() + "_deleted", "on_" + param->GetName() + "_modified", update_slot->GetSignature(), this);
        }
    }

    if(firstInit == false)
    {
		auto connections_copy = _pimpl->_parameter_connections;
		_pimpl->_parameter_connections.clear();
		for (auto& parameter_connection : connections_copy)
		{
			rcc::shared_ptr<IMetaObject> obj(parameter_connection.output_object);
			if (obj)
			{
				auto output = obj->GetOutput(parameter_connection.output_parameter);
				if (output == nullptr)
				{
                    LOG(debug) << "Unable to find " << parameter_connection.output_parameter
                               << " in " << obj->GetTypeName() << " reinitializing";
					obj->InitParameters(firstInit);
					output = obj->GetOutput(parameter_connection.output_parameter);
					if (output == nullptr)
                    {
                        LOG(info) << "Unable to find " << parameter_connection.output_parameter << " in "
                                  << obj->GetTypeName() << " unable to reconnect " << parameter_connection.input_parameter
                                  << " from object " << this->GetTypeName();
						continue;
                    }
				}
				auto input = this->GetInput(parameter_connection.input_parameter);
				if (input)
				{
                    if(this->ConnectInput(input, obj.Get(), output, parameter_connection.connection_type))
                    {
                       LOG(debug) << "Reconnected " << GetTypeName() << ":" << parameter_connection.input_parameter
                                  << " to " << obj->GetTypeName() << ":" << parameter_connection.output_parameter;
                    }else
                    {
                        LOG(info) << "Reconnect FAILED " << GetTypeName() << ":" << parameter_connection.input_parameter
                                   << " to " << obj->GetTypeName() << ":" << parameter_connection.output_parameter;
                    }
				}
				else
				{
                    LOG(debug) << "Unable to find input parameter "
                               << parameter_connection.input_parameter
                               << " in object " << this->GetTypeName();
				}
            }else
            {
                LOG(debug) << "Output object no longer exists for input ["
                           << parameter_connection.input_parameter
                           << "] expected output name ["
                           << parameter_connection.output_parameter << "]";
            }
		}
        // Rebuild connections
        for(auto& connection : _pimpl->_connections)
        {
            if(!connection.obj.empty())
            {
                auto signal = this->GetSignal(connection.signal_name, connection.signature);
                auto slot = connection.obj->GetSlot(connection.slot_name, connection.signature);
                if(signal == nullptr)
                {
                    LOG(debug) << "Unable to find signal with name \"" << connection.signal_name
                               << "\" and signature: " << connection.signature.name()
                               << " in new object of type " << this->GetTypeName();
                }
                if(slot == nullptr)
                {
                    connection.obj->BindSlots(firstInit);
                    slot = connection.obj->GetSlot(connection.slot_name, connection.signature);
                    if(slot == nullptr)
                    {
                        LOG(debug) << "Unable to find slot with name \"" << connection.slot_name
                                   << "\" and signature: " << connection.signature.name()
                                   << " in new object of type " << connection.obj->GetTypeName();
                    }
                }
                if(signal && slot)
                {
                    auto connection_ = slot->Connect(signal);
                    if (connection_)
                    {
                        connection.connection = connection_;
                    }
                }
            }
        }
    }
}
void  IMetaObject::InitCustom(bool firstInit)
{

}

int IMetaObject::SetupSignals(RelayManager* manager)
{
    _sig_manager = manager;
    int count = 0;
    for(auto& my_slots : _pimpl->_slots)
    {
        for(auto& slot : my_slots.second)
        {
            ConnectionInfo info;
            info.connection = manager->Connect(slot.second, my_slots.first, this);
            info.slot_name = my_slots.first;
            info.signature = slot.first;
            _pimpl->_connections.push_back(info);
            ++count;
        }
    }

    for(auto& my_signals : _pimpl->_signals)
    {
        for(auto& signal : my_signals.second)
        {
            auto connection = manager->Connect(signal.second, my_signals.first, this);
            ConnectionInfo info;
            info.signal_name = my_signals.first;
            info.signature = signal.first;
            info.connection = connection;
            _pimpl->_connections.push_back(info);
            ++count;
        }
    }

    return count;
}

int IMetaObject::SetupVariableManager(IVariableManager* manager)
{
    if(_pimpl->_variable_manager != nullptr)
    {
        RemoveVariableManager(_pimpl->_variable_manager);
    }
    _pimpl->_variable_manager = manager;
    int count = 0;
    for(auto& param : _pimpl->_implicit_parameters)
    {
        manager->AddParameter(param.second.get());
        ++count;
    }
    for(auto& param : _pimpl->_parameters)
    {
        manager->AddParameter(param.second);
        ++count;
    }
    return count;
}

int IMetaObject::RemoveVariableManager(IVariableManager* mgr)
{
    int count = 0;
    for (auto& param : _pimpl->_implicit_parameters)
    {
        mgr->RemoveParameter(param.second.get());
        ++count;
    }
    for (auto& param : _pimpl->_parameters)
    {
        mgr->RemoveParameter(param.second);
        ++count;
    }
    return count;
}

void IMetaObject::Serialize(ISimpleSerializer *pSerializer)
{
    IObject::Serialize(pSerializer);
    SerializeConnections(pSerializer);
    SerializeParameters(pSerializer);
}

void IMetaObject::SerializeConnections(ISimpleSerializer* pSerializer)
{
    SERIALIZE(_pimpl->_connections);
	SERIALIZE(_pimpl->_parameter_connections);
    SERIALIZE(_ctx);
    SERIALIZE(_sig_manager);
}
void IMetaObject::SerializeParameters(ISimpleSerializer* pSerializer)
{

}
void IMetaObject::SetContext(Context* ctx, bool overwrite)
{
    if(_ctx && overwrite == false)
        return;
    if(ctx == nullptr)
        LOG(info) << "Setting context to nullptr";
    _ctx = ctx;
    for(auto& param : _pimpl->_implicit_parameters)
    {
        param.second->SetContext(ctx);
    }
    for(auto& param : _pimpl->_parameters)
    {
        param.second->SetContext(ctx);
    }
}

int IMetaObject::DisconnectByName(const std::string& name)
{
    auto my_signals = this->GetSignals(name);
    int count = 0;
    for(auto& sig : my_signals)
    {
		count += sig->Disconnect() ? 1 : 0;
    }
    return count;
}

bool IMetaObject::Disconnect(ISignal* sig)
{
    return false;
}

int IMetaObject::Disconnect(IMetaObject* obj)
{
    auto obj_signals = obj->GetSignals();
    int count = 0;
    for(auto signal : obj_signals)
    {
        count += Disconnect(signal.first) ? 1 : 0;
    }
    return count;
}


std::vector<IParameter*> IMetaObject::GetDisplayParameters() const
{
    std::vector<IParameter*> output;
    for(auto& param : _pimpl->_parameters)
    {
        output.push_back(param.second);
    }
    for(auto& param : _pimpl->_implicit_parameters)
    {
        output.push_back(param.second.get());
    }
    return output;
}

IParameter* IMetaObject::GetParameter(const std::string& name) const
{
    auto itr = _pimpl->_parameters.find(name);
    if(itr != _pimpl->_parameters.end())
    {
        return itr->second;
    }
    auto itr2 = _pimpl->_implicit_parameters.find(name);
    if(itr2 != _pimpl->_implicit_parameters.end())
    {
        return itr2->second.get();
    }
    THROW(debug) << "Parameter with name \"" << name << "\" not found";
    return nullptr;
}
std::vector<IParameter*> IMetaObject::GetParameters(const std::string& filter) const
{
    std::vector<IParameter*> output;
    for(auto& itr : _pimpl->_parameters)
    {
        if(filter.size())
        {
            if(itr.first.find(filter) != std::string::npos)
                output.push_back(itr.second);
        }else
        {
            output.push_back(itr.second);
        }
    }
    for(auto& itr : _pimpl->_implicit_parameters)
    {
        if (filter.size())
        {
            if (itr.first.find(filter) != std::string::npos)
                output.push_back(itr.second.get());
        }
        else
        {
            output.push_back(itr.second.get());
        }
    }
    return output;
}

std::vector<IParameter*> IMetaObject::GetParameters(const TypeInfo& filter) const
{
    std::vector<IParameter*> output;
    for (auto& itr : _pimpl->_parameters)
    {
        if(itr.second->GetTypeInfo() == filter)
            output.push_back(itr.second);
        
    }
    for (auto& itr : _pimpl->_implicit_parameters)
    {
        if(itr.second->GetTypeInfo() == filter)
            output.push_back(itr.second.get());
    }
    return output;
}

IParameter* IMetaObject::GetParameterOptional(const std::string& name) const
{
    auto itr = _pimpl->_parameters.find(name);
    if(itr != _pimpl->_parameters.end())
    {
        return itr->second;
    }
    auto itr2 = _pimpl->_implicit_parameters.find(name);
    if(itr2 != _pimpl->_implicit_parameters.end())
    {
        return itr2->second.get();
    }
    LOG(trace) << "Parameter with name \"" << name << "\" not found";
    return nullptr;
}

InputParameter* IMetaObject::GetInput(const std::string& name) const
{
    auto itr = _pimpl->_input_parameters.find(name);
    if(itr != _pimpl->_input_parameters.end())
    {
        return itr->second;   
    }
    return nullptr;
}

Context* IMetaObject::GetContext() const
{
    return _ctx;
}

std::vector<InputParameter*> IMetaObject::GetInputs(const std::string& name_filter) const
{
    std::vector<InputParameter*> output;
    for(auto param : _pimpl->_input_parameters)
    {
        if(name_filter.size())
        {
            if(param.second->GetName().find(name_filter) != std::string::npos)
                output.push_back(param.second);
        }else
        {
            output.push_back(param.second);
        }
        
    }
    return output;
}

std::vector<InputParameter*> IMetaObject::GetInputs(const TypeInfo& type_filter, const std::string& name_filter) const
{
    std::vector<InputParameter*> output;
    for(auto param : _pimpl->_parameters)
    {
        if(param.second->CheckFlags(Input_e))
        {
            if(param.second->GetTypeInfo() == type_filter)
            {
                if(name_filter.size())
                {
                    if(name_filter.find(param.first) != std::string::npos)
                        if(auto out = dynamic_cast<InputParameter*>(param.second))
                            output.push_back(out);
                }else
                {
                    if(auto out = dynamic_cast<InputParameter*>(param.second))
                        output.push_back(out);
                }
            }
        }
    }
    for(auto param : _pimpl->_implicit_parameters)
    {
        if(param.second->CheckFlags(Input_e))
        {
            if(param.second->GetTypeInfo() == type_filter)
            {
                if(name_filter.size())
                {
                    if(name_filter.find(param.first) != std::string::npos)
                        if(auto out = dynamic_cast<InputParameter*>(param.second.get()))
                            output.push_back(out);
                }else
                {
                    if(auto out = dynamic_cast<InputParameter*>(param.second.get()))
                        output.push_back(out);
                }
            }
        }
    }
    return output;
}

IParameter* IMetaObject::GetOutput(const std::string& name) const
{
    auto itr = _pimpl->_parameters.find(name);
    if(itr != _pimpl->_parameters.end())
    {
        return itr->second;
    }
    auto itr2 = _pimpl->_implicit_parameters.find(name);
    if(itr2 != _pimpl->_implicit_parameters.end())
    {
        return itr2->second.get();
    }
    return nullptr;
}
std::vector<IParameter*> IMetaObject::GetOutputs(const std::string& name_filter) const
{
    std::vector<IParameter*> output;
    for(auto param : _pimpl->_parameters)
    {
        if(param.second->CheckFlags(Output_e))
        {
            if(name_filter.size())
            {
                if(param.first.find(name_filter) != std::string::npos)
                {
                    output.push_back(param.second);
                }
            }else
            {
                output.push_back(param.second);
            }
        }
    }
    for(auto param : _pimpl->_implicit_parameters)
    {
        if(param.second->CheckFlags(Output_e))
        {
            if(name_filter.size())
            {
                if(param.first.find(name_filter) != std::string::npos)
                {
                    output.push_back(param.second.get());
                }
            }else
            {
                output.push_back(param.second.get());
            }
        }
    }
    return output;
}

std::vector<IParameter*> IMetaObject::GetOutputs(const TypeInfo& type_filter, const std::string& name_filter) const
{
    std::vector<IParameter*> output;
    for(auto param : _pimpl->_parameters)
    {
        if(param.second->CheckFlags(Output_e))
        {
            if(name_filter.size())
            {
                if(name_filter.find(param.first) != std::string::npos)
                {
                    if(param.second->GetTypeInfo() == type_filter)
                        output.push_back(param.second);
                }
            }else
            {
                if(param.second->GetTypeInfo() == type_filter)
                    output.push_back(param.second);
            }
        }
    }
    for(auto param : _pimpl->_implicit_parameters)
    {
        if(param.second->CheckFlags(Output_e))
        {
            if(name_filter.size())
            {
                if(name_filter.find(param.first) != std::string::npos)
                {
                    if(param.second->GetTypeInfo() == type_filter)
                        output.push_back(param.second.get());
                }
            }else
            {
                if(param.second->GetTypeInfo() == type_filter)
                    output.push_back(param.second.get());
            }
        }
    }
    return output;
}
bool IMetaObject::ConnectInput(const std::string& input_name,
                               IMetaObject* output_object,
                               IParameter* output,
                               ParameterTypeFlags type_)
{
    auto input = GetInput(input_name);
    if(input && output)
        return ConnectInput(input, output_object, output, type_);

    auto inputs = GetInputs();
    auto print_inputs = [inputs]()->std::string
    {
        std::stringstream ss;
        for(auto _input : inputs)
        {
            ss << dynamic_cast<IParameter*>(_input)->GetName() << ", ";
        }
        return ss.str();
    };
    LOG(debug) << "Unable to find input by name "
               << input_name << " in object "
               << this->GetTypeName()
               << " with inputs [" << print_inputs() << "]";
    return false;
}

bool IMetaObject::ConnectInput(InputParameter* input,
                               IMetaObject* output_object,
                               IParameter* output,
                               ParameterTypeFlags type_)
{
    if(input == nullptr || output == nullptr)
    {
        LOG(debug) << "NULL input or output passed in";
        return false;
    }
    
    if(input && input->AcceptsInput(output))
    {
        // Check contexts to see if a buffer needs to be setup
        auto output_ctx = output->GetContext();
        if(type_ & ForceBufferedConnection_e)
        {
            type_ = ParameterTypeFlags(type_ & ~ForceBufferedConnection_e);
            auto buffer = Buffer::BufferFactory::CreateProxy(output, type_);
            if(!buffer)
            {
                LOG(warning) << "Unable to create " << ParameterTypeFlagsToString(type_)
                             << " for datatype " << Demangle::TypeToName(output->GetTypeInfo());
                return false;
            }
            std::string buffer_type = ParameterTypeFlagsToString(type_);
            buffer->SetName(output->GetTreeName() + " " + buffer_type + " buffer for " + input->GetTreeName());
            if(input->SetInput(buffer))
            {
                _pimpl->_parameter_connections.emplace_back(output_object, output->GetName(), input->GetName(), type_);
                return true;
            }
            else
            {
                LOG(debug) << "Failed to connect output " << output->GetName()
                           << "[" << Demangle::TypeToName(output->GetTypeInfo())<< "] to input "
                           << dynamic_cast<IParameter*>(input)->GetName()
                            << "[" << Demangle::TypeToName(dynamic_cast<IParameter*>(input)->GetTypeInfo()) << "]";
                return false;
            }
        }
        if(output_ctx && _ctx)
        {
            if(output_ctx->thread_id != _ctx->thread_id)
            {
                auto buffer = Buffer::BufferFactory::CreateProxy(output, type_);
                if(buffer)
                {
                    buffer->SetName(output->GetTreeName() + " buffer for " + input->GetTreeName());
                    if(input->SetInput(buffer))
					{
						_pimpl->_parameter_connections.emplace_back(output_object, output->GetName(), input->GetName(), type_);
						return true;
					}
                    else
                    {
                        LOG(debug) << "Failed to connect output " << output->GetName()
                                   << "[" << Demangle::TypeToName(output->GetTypeInfo()) << "] to input "
                            << dynamic_cast<IParameter*>(input)->GetName() << "["
                            << Demangle::TypeToName(dynamic_cast<IParameter*>(input)->GetTypeInfo()) << "]";
						return false;
                    }
                }else
                {
                    LOG(debug) << "No buffer of desired type found for type " << Demangle::TypeToName(output->GetTypeInfo());
                }
            }else
            {
                if(input->SetInput(output))
				{
					_pimpl->_parameter_connections.emplace_back(output_object, output->GetName(), input->GetName(), type_);
					return true;
				}
                else
                {
                    LOG(debug) << "Failed to connect output " << output->GetName()
                               << "[" << Demangle::TypeToName(output->GetTypeInfo()) << "] to input "
                               << dynamic_cast<IParameter*>(input)->GetName()
                               << "[" << Demangle::TypeToName(dynamic_cast<IParameter*>(input)->GetTypeInfo()) << "]";
					return false;
                }
            }
        }else
        {
            if(input->SetInput(output))
			{
				_pimpl->_parameter_connections.emplace_back(output_object, output->GetName(), input->GetName(), type_);
				return true;
			}else
            {
                LOG(debug) << "Failed to connect output " << output->GetName()
                           << "[" << Demangle::TypeToName(output->GetTypeInfo()) << "] to input "
                           << dynamic_cast<IParameter*>(input)->GetName()
                           << "[" << Demangle::TypeToName(dynamic_cast<IParameter*>(input)->GetTypeInfo()) << "]";
				return false;
            }
        }
    }
    LOG(debug) << "Input \"" << input->GetTreeName()
               << "\"  does not accept input of type: "
               << Demangle::TypeToName(output->GetTypeInfo());
    return false;
}

bool IMetaObject::ConnectInput(IMetaObject* out_obj, IParameter* out_param, 
                               IMetaObject* in_obj, InputParameter* in_param,
                               ParameterTypeFlags type)
{
	return in_obj->ConnectInput(in_param, out_obj, out_param, type);
}								 
IParameter* IMetaObject::AddParameter(std::shared_ptr<IParameter> param)
{
    param->SetMtx(_mtx);
    param->SetContext(_ctx);
#ifdef _DEBUG
    for(auto& param_ : _pimpl->_parameters)
    {
        if(param_.second == param.get())
        {
            LOG(debug) << "Trying to add a parameter a second time";
            return param.get();
        }
    }
#endif
    _pimpl->_implicit_parameters[param->GetName()] = param;
    if(param->CheckFlags(Input_e))
    {
        _pimpl->_input_parameters[param->GetName()] = dynamic_cast<InputParameter*>(param.get());
    }
    param->RegisterUpdateNotifier(&(this->_pimpl->_slot_parameter_updated));
    _pimpl->_sig_parameter_added(this, param.get());
    return param.get();
}

IParameter* IMetaObject::AddParameter(IParameter* param)
{
    param->SetMtx(_mtx);
    param->SetContext(_ctx);
#ifdef _DEBUG
    for(auto& param_ : _pimpl->_parameters)
    {
        if(param_.second == param)
        {
            LOG(debug) << "Trying to add a parameter a second time";
            return param;
        }
    }
#endif
    _pimpl->_parameters[param->GetName()] = param;
    if(param->CheckFlags(Input_e))
    {
        _pimpl->_input_parameters[param->GetName()] = dynamic_cast<InputParameter*>(param);
    }
    auto connection = param->RegisterUpdateNotifier(&(this->_pimpl->_slot_parameter_updated));
    _pimpl->_sig_parameter_added(this, param);
    this->AddConnection(connection, "parameter_update", "parameter_updated",
                        this->_pimpl->_slot_parameter_updated.GetSignature(), this);
    return param;
}

void IMetaObject::SetParameterRoot(const std::string& root)
{
    for(auto& param : _pimpl->_parameters)
    {
        param.second->SetTreeRoot(root);
    }
    for(auto& param : _pimpl->_implicit_parameters)
    {
        param.second->SetTreeRoot(root);
    }
}

std::vector<ParameterInfo*> IMetaObject::GetParameterInfo(const std::string& name) const
{
    std::vector<ParameterInfo*> output;
    GetParameterInfo(output);

    return output;
}

std::vector<ParameterInfo*> IMetaObject::GetParameterInfo() const
{
    std::vector<ParameterInfo*> output;
    GetParameterInfo(output);

    return output;
}

std::vector<SignalInfo*>    IMetaObject::GetSignalInfo(const std::string& name) const
{
    std::vector<SignalInfo*> info;
    GetSignalInfo(info);

    return info;
}
std::vector<SignalInfo*>    IMetaObject::GetSignalInfo() const
{
    std::vector<SignalInfo*> info;
    GetSignalInfo(info);
    return info;
}
std::vector<SlotInfo*> IMetaObject::GetSlotInfo() const
{
    std::vector<SlotInfo*> output;
    GetSlotInfo(output);
    return output;
}
std::vector<SlotInfo*> IMetaObject::GetSlotInfo(const std::string& name) const
{
    std::vector<SlotInfo*> tmp;
    GetSlotInfo(tmp);
    std::vector<SlotInfo*> output;
    for (auto& itr : tmp)
    {
        if (itr->name.find(name) != std::string::npos)
            output.push_back(itr);
    }

    return output;
}

std::vector<std::pair<ISlot*, std::string>>  IMetaObject::GetSlots() const
{
    std::vector<std::pair<ISlot*, std::string>>  my_slots;
    for(auto itr1 : _pimpl->_slots)
    {
        for(auto itr2: itr1.second)
        {
            my_slots.push_back(std::make_pair(itr2.second, itr1.first));
        }
    }
    return my_slots;
}

std::vector<ISlot*> IMetaObject::GetSlots(const std::string& name) const
{
    std::vector<ISlot*> output;
    auto itr = _pimpl->_slots.find(name);
    if(itr != _pimpl->_slots.end())
    {
        for(auto slot : itr->second)
        {
            output.push_back(slot.second);
        }
    }
    return output;
}

std::vector<std::pair<ISlot*, std::string>> IMetaObject::GetSlots(const TypeInfo& signature) const
{
	std::vector<std::pair<ISlot*, std::string>> output;
    for(auto& type : _pimpl->_slots)
    {
        auto itr = type.second.find(signature);
        if(itr != type.second.end())
        {
            output.push_back(std::make_pair(itr->second, type.first));
        }
    }
    return output;
}

ISlot* IMetaObject::GetSlot(const std::string& name, const TypeInfo& signature) const
{
    auto itr1 = _pimpl->_slots.find(name);
    if(itr1 != _pimpl->_slots.end())
    {
        auto itr2 = itr1->second.find(signature);
        if(itr2 != itr1->second.end())
        {
            return itr2->second;
        }
    }
    if(name == "parameter_updated")
    {
        return &(_pimpl->_slot_parameter_updated);
    }
    return nullptr;
}

bool IMetaObject::ConnectByName(const std::string& name, ISlot* slot)
{
	auto signal = GetSignal(name, slot->GetSignature());
	if (signal)
	{
		auto connection = signal->Connect(slot);
		if (connection)
		{
			AddConnection(connection, name, "", slot->GetSignature());
			return true;
		}
	}
	return false;
}
bool IMetaObject::ConnectByName(const std::string& name, ISignal* signal)
{
	auto slot = GetSlot(name, signal->GetSignature());
	if (slot)
	{
		auto connection = slot->Connect(signal);
		if (connection)
		{
			AddConnection(connection, "", name, signal->GetSignature());
			return true;
		}
	}
	return false;
}

int IMetaObject::ConnectByName(const std::string& name, RelayManager* mgr)
{

	return 0;
}

int  IMetaObject::ConnectByName(const std::string& signal_name,
                                IMetaObject* receiver,
                                const std::string& slot_name)
{
	int count = 0;
    auto my_signals = GetSignals(signal_name);
    auto my_slots = receiver->GetSlots(slot_name);
    for (auto signal : my_signals)
	{
        for (auto slot : my_slots)
		{
			if (signal->GetSignature() == slot->GetSignature())
			{
				auto connection = slot->Connect(signal);
				if (connection)
				{
                    AddConnection(connection, signal_name,
                                  slot_name, slot->GetSignature(), receiver);
					++count;
					break;
				}
			}
		}
	}
	return count;
}

bool IMetaObject::ConnectByName(const std::string& signal_name,
                                IMetaObject* receiver,
                                const std::string& slot_name,
                                const TypeInfo& signature)
{
	auto signal = GetSignal(signal_name, signature);
	auto slot = receiver->GetSlot(slot_name, signature);
	if (signal && slot)
	{
		auto connection = slot->Connect(signal);
		if (connection)
		{
			AddConnection(connection, signal_name, slot_name, signature, receiver);
			return true;
		}
	}
	return false;
}

int IMetaObject::ConnectAll(RelayManager* mgr)
{
    auto my_signals = GetSignalInfo();
    int count = 0;
    for(auto& signal : my_signals)
    {
        count += ConnectByName(signal->name, mgr);
    }
    return count;
}



void IMetaObject::AddSlot(ISlot* slot, const std::string& name)
{
    _pimpl->_slots[name][slot->GetSignature()] = slot;
	slot->SetParent(this);
}

void IMetaObject::AddSignal(ISignal* sig, const std::string& name)
{
	_pimpl->_signals[name][sig->GetSignature()] = sig;
	sig->SetParent(this);
}

std::vector<std::pair<ISignal*, std::string>> IMetaObject::GetSignals() const
{
    std::vector<std::pair<ISignal*, std::string>> my_signals;
    for(auto& name_itr : _pimpl->_signals)
    {
        for(auto& sig_itr : name_itr.second)
        {
            my_signals.push_back(std::make_pair(sig_itr.second, name_itr.first));
        }
    }
    return my_signals;
}

std::vector<ISignal*> IMetaObject::GetSignals(const std::string& name) const
{
    std::vector<ISignal*> my_signals;
    auto itr = _pimpl->_signals.find(name);
    if(itr != _pimpl->_signals.end())
    {
        for(auto& sig_itr : itr->second)
        {
            my_signals.push_back(sig_itr.second);
        }
    }
    return my_signals;
}

std::vector<std::pair<ISignal*, std::string>> IMetaObject::GetSignals(const TypeInfo& type) const
{
    std::vector<std::pair<ISignal*, std::string>> my_signals;
    for(auto& name_itr : _pimpl->_signals)
    {
        auto type_itr = name_itr.second.find(type);
        if(type_itr != name_itr.second.end())
        {
            my_signals.push_back(std::make_pair(type_itr->second, name_itr.first));
        }
    }
    return my_signals;
}

ISignal* IMetaObject::GetSignal(const std::string& name, const TypeInfo& type) const
{
	auto name_itr = _pimpl->_signals.find(name);
	if (name_itr != _pimpl->_signals.end())
	{
		auto type_itr = name_itr->second.find(type);
		if (type_itr != name_itr->second.end())
		{
			return type_itr->second;
		}
	}
    if(name == "parameter_updated")
    {
        return &(_pimpl->_sig_parameter_updated);
    }
    if(name == "parameter_added")
    {
        return &(_pimpl->_sig_parameter_added);
    }
	return nullptr;
}

void IMetaObject::AddConnection(std::shared_ptr<Connection>& connection,
                                const std::string& signal_name,
                                const std::string& slot_name,
                                const TypeInfo& signature,
                                IMetaObject* obj)
{
	ConnectionInfo info;
	info.connection = connection;
	info.obj = rcc::weak_ptr<IMetaObject>(obj);
	info.signal_name = signal_name;
	info.slot_name = slot_name;
	info.signature = signature;
	_pimpl->_connections.push_back(info);
}
void IMetaObject::onParameterUpdate(Context* ctx, IParameter* param)
{
    this->_pimpl->_sig_parameter_updated(this, param);
}
