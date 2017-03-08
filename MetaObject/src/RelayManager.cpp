#include "MetaObject/Signals/RelayManager.hpp"
#include "MetaObject/Signals/ISignalRelay.hpp"
#include "MetaObject/Signals/RelayFactory.hpp"
#include "MetaObject/Signals/ISlot.hpp"
#include "MetaObject/Signals/ISignal.hpp"
#include "MetaObject/IMetaObject.hpp"
#include <map>
#include <memory>
using namespace mo;

struct RelayManager::impl
{
	std::map<TypeInfo, std::map<std::string, std::shared_ptr<ISignalRelay>>> relays;
};

RelayManager::RelayManager()
{
	_pimpl = new impl();
}

RelayManager::~RelayManager()
{
	delete _pimpl;
}

RelayManager* g_inst = nullptr;
RelayManager* RelayManager::Instance()
{
	if (g_inst == nullptr)
		g_inst = new RelayManager();
	return g_inst;
}

void RelayManager::SetInstance(RelayManager* inst)
{
	g_inst = inst;
}

std::shared_ptr<Connection> RelayManager::Connect(ISlot* slot, const std::string& name, IMetaObject* obj)
{
	auto& relay = GetRelay(slot->GetSignature(), name);
	return slot->Connect(relay);
}

std::shared_ptr<Connection> RelayManager::Connect(ISignal* signal, const std::string& name, IMetaObject* obj)
{
	auto& relay = GetRelay(signal->GetSignature(), name);
	return signal->Connect(relay);
}

void RelayManager::ConnectSignal(IMetaObject* obj, const std::string& signal_name)
{
    auto signals = obj->GetSignals(signal_name);
    for(auto signal : signals)
    {
        auto connection = Connect(signal, signal_name, obj);
        if(connection)
        {
            obj->AddConnection(connection, signal_name, signal_name, signal->GetSignature(), nullptr);
        }
    }
}

void RelayManager::ConnectSlot(IMetaObject* obj, const std::string& slot_name)
{
    auto slots = obj->GetSlots(slot_name);
    for (auto slot : slots)
    {
        auto connection = Connect(slot, slot_name, obj);
        if (connection)
        {
            obj->AddConnection(connection, slot_name, slot_name, slot->GetSignature(), nullptr);
        }
    }
}

bool RelayManager::ConnectSignal(IMetaObject* obj, const std::string& name, const TypeInfo& type)
{
	auto signal = obj->GetSignal(name, type);
	if (signal)
	{
		auto connection = Connect(signal, name, obj);
		if (connection)
		{
			obj->AddConnection(connection, name, "", signal->GetSignature());
			return true;
		}
	}
	return false;
}
bool RelayManager::ConnectSlot(IMetaObject* obj, const std::string& name, const TypeInfo& type)
{
	auto slot = obj->GetSlot(name, type);
	if (slot)
	{
		auto connection = Connect(slot, name, obj);
		if (connection)
		{
			obj->AddConnection(connection, "", name, type, nullptr);
		}
	}
	return false;
}

int RelayManager::ConnectSignals(IMetaObject* obj, const std::string& name)
{
	int count = 0;
	auto signals = obj->GetSignals(name);
	for (auto signal : signals)
	{
		count += Connect(signal, name, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::ConnectSignals(IMetaObject* obj, const TypeInfo& type)
{
	int count = 0;
	auto signals = obj->GetSignals(type);
	for (auto signal : signals)
	{
		count += Connect(signal.first, signal.second, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::ConnectSignals(IMetaObject* obj)
{
	int count = 0;
	auto signals = obj->GetSignals();
	for (auto signal : signals)
	{
		count += Connect(signal.first, signal.second, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::ConnectSlots(IMetaObject* obj, const std::string& name)
{
	int count = 0;
	auto slots = obj->GetSlots(name);
	for (auto& slot : slots)
	{
		count += Connect(slot, name, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::ConnectSlots(IMetaObject* obj, const TypeInfo& type)
{
	int count = 0;
    auto all_slots = obj->GetSlots(type);
    for (auto& slot : all_slots)
	{
		count += Connect(slot.first, slot.second, obj) ? 1 : 0;
	}
	return count;
}

int RelayManager::ConnectSlots(IMetaObject* obj)
{
	int count = 0;
    auto all_slots = obj->GetSlots();
    for (auto& slot : all_slots )
	{
		count += Connect(slot.first, slot.second, obj) ? 1 : 0;
	}
	return count;
}

std::vector<std::shared_ptr<ISignalRelay>> RelayManager::GetRelays(const std::string& name)
{
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<std::shared_ptr<ISignalRelay>> relays;
    for(auto& types : _pimpl->relays)
    {
        if(name.size())
        {
            auto itr = types.second.find(name);
            if(itr != types.second.end())
            {
                relays.push_back(itr->second);
            }
        }else
        {
            for(auto& relay : types.second)
            {
                relays.push_back(relay.second);
            }
        }
    }
    return relays;
}
std::vector<std::pair<std::shared_ptr<ISignalRelay>, std::string>> RelayManager::GetAllRelays()
{
    std::lock_guard<std::mutex> lock(mtx);
    std::vector<std::pair<std::shared_ptr<ISignalRelay>, std::string>> output;
    for(auto& types : _pimpl->relays)
    {
        for(auto& relay : types.second)
        {
            output.emplace_back(relay.second, relay.first);
        }
    }
    return output;
}

std::shared_ptr<ISignalRelay>& RelayManager::GetRelay(const TypeInfo& type, const std::string& name)
{
    std::lock_guard<std::mutex> lock(mtx);
	return _pimpl->relays[type][name];
}

bool RelayManager::exists(const std::string& name, TypeInfo type)
{
	auto itr1 = _pimpl->relays.find(type);
	if (itr1 != _pimpl->relays.end())
	{
		auto itr2 = itr1->second.find(name);
		if (itr2 != itr1->second.end())
		{
			return true;
		}
	}
	return false;
}
