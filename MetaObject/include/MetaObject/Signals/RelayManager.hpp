#pragma once
#include "MetaObject/Detail/Export.hpp"
#include "MetaObject/Detail/TypeInfo.h"
#include <mutex>
#include <memory>
#include <vector>

namespace mo
{
    class ISignal;
	class ISlot;
    class IMetaObject;
	class ISignalRelay;
	class Connection;
	template<typename T> class TypedSignal;
	template<typename T> class TypedSlot;
	template<typename T> class TypedSignalRelay;
    
    // Manages shared ownership of signals so that multiple senders and receivers can exist, also allows for encapsulation of groups of signals based on subsystem
    class MO_EXPORTS RelayManager
    {
    public:
		RelayManager();
        ~RelayManager();
        static RelayManager* Instance();
        static void SetInstance(RelayManager* inst);
		
		std::shared_ptr<Connection> Connect(ISlot* slot, const std::string& name, IMetaObject* obj = nullptr);
		std::shared_ptr<Connection> Connect(ISignal* signal, const std::string& name, IMetaObject* obj = nullptr);
        void ConnectSignal(IMetaObject* obj, const std::string& signal_name);
        void ConnectSlot(IMetaObject* obj, const std::string& slot_name);

		bool ConnectSignal(IMetaObject* obj, const std::string& name, const TypeInfo& type);
		int  ConnectSignals(IMetaObject* obj, const std::string& name);
		int  ConnectSignals(IMetaObject* obj, const TypeInfo& type);
		int  ConnectSignals(IMetaObject* obj);

		bool ConnectSlot(IMetaObject* obj, const std::string& name, const TypeInfo& type);
		int  ConnectSlots(IMetaObject* obj, const std::string& name);
		int  ConnectSlots(IMetaObject* obj, const TypeInfo& type);
		int  ConnectSlots(IMetaObject* obj);

        
        std::vector<std::shared_ptr<ISignalRelay>> GetRelays(const std::string& name);
        std::vector<std::pair<std::shared_ptr<ISignalRelay>, std::string>> GetAllRelays();
        template<class Sig> std::shared_ptr<TypedSignalRelay<Sig>> GetRelay(const std::string& name)
        {
            return std::dynamic_pointer_cast<TypedSignalRelay<Sig>>(GetRelay(TypeInfo(typeid(Sig)), name));
        }
    protected:
        std::shared_ptr<ISignalRelay>& GetRelay(const TypeInfo& type, const std::string& name);
        
        bool exists(const std::string& name, TypeInfo type);
    private:
        struct impl;
        impl* _pimpl;
        std::mutex mtx;
    };
} // namespace Signals
