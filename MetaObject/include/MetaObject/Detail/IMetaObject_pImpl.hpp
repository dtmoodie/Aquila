#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "shared_ptr.hpp"
#include <string>
#include <memory>
#include <map>
#include <set>
#include <list>
namespace mo
{
    class ICallback;
    class Connection;
    class ISignal;
    class ISlot;
    class IParameter;
	class IMetaObject;
	struct MO_EXPORTS ConnectionInfo
	{
        ConnectionInfo()
        {
        }
        ConnectionInfo(const ConnectionInfo& info)
        {
            signal_name = info.signal_name;
            slot_name = info.slot_name;
            signature = info.signature;
            obj = info.obj;
            connection = info.connection;
        }
		std::string signal_name;
		std::string slot_name;
		TypeInfo signature;
		rcc::weak_ptr<IMetaObject> obj;
		std::shared_ptr<Connection> connection;
	};
	struct MO_EXPORTS ParameterConnectionInfo
	{
		ParameterConnectionInfo(const rcc::weak_ptr<IMetaObject>& obj, const std::string& output, const std::string& input, ParameterTypeFlags type) :
			output_object(obj), output_parameter(output), input_parameter(input), connection_type(type)
		{
		}
		rcc::weak_ptr<IMetaObject> output_object;
		std::string output_parameter;
		std::string input_parameter;
		ParameterTypeFlags connection_type;
	};
    struct MO_EXPORTS IMetaObject::impl
    {
        impl()
        {
            _variable_manager = nullptr;
            _signals["parameter_updated"][_sig_parameter_updated.GetSignature()] = &_sig_parameter_updated;
            _signals["parameter_added"][_sig_parameter_updated.GetSignature()] = &_sig_parameter_added;
        }
        std::map<std::string, std::map<TypeInfo, ISignal*>> _signals;
        std::map<std::string, std::map<TypeInfo, ISlot*>>   _slots;

        std::map<std::string, IParameter*>				    _parameters; // statically defined in object

        std::map<std::string, std::shared_ptr<IParameter>>  _implicit_parameters; // Can be changed at runtime
		std::list<ConnectionInfo> _connections;
        std::list<ParameterConnectionInfo> _parameter_connections;

        TypedSignal<void(IMetaObject*, IParameter*)> _sig_parameter_updated;
        TypedSignal<void(IMetaObject*, IParameter*)> _sig_parameter_added;
        std::map<std::string, InputParameter*>       _input_parameters;
        TypedSlot<void(Context* ctx, IParameter*)>   _slot_parameter_updated;
        IVariableManager*                            _variable_manager;
    };
}