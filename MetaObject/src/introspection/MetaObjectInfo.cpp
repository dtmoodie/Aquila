#include "MetaObject/IMetaObjectInfo.hpp"
#include "MetaObject/Parameters/ParameterInfo.hpp"
#include "MetaObject/Parameters/Demangle.hpp"
#include "MetaObject/Signals/SignalInfo.hpp"
#include "MetaObject/Signals/SlotInfo.hpp"

#include <sstream>
using namespace mo;

std::string IMetaObjectInfo::Print() const
{
    std::stringstream ss;
    ss << "\n\n";
    std::string name = GetObjectName();

    ss << GetInterfaceId(); // << " *** " << GetObjectName() << " ***\n";
    ss << " ***** ";
    ss << name << " ";
    if(name.size() < 20)
        for(int i = 0; i < 20 - name.size(); ++i)
            ss << "*";
    ss << "\n";

    auto tooltip = GetObjectTooltip();
    if(tooltip.size())
        ss << "  " << tooltip << "\n";
    auto help = GetObjectHelp();
    if(help.size())
        ss << "    " << help << "\n";
    auto params = GetParameterInfo();
    if(params.size())
    {
        ss << "----------- Parameters ------------- \n";
        int longest_name = 0;
        for(auto& slot : params)
            longest_name = std::max<int>(longest_name, slot->name.size());
        longest_name += 1;
        for(auto& param : params)
        {
            ss <<  param->name;
            for(int i = 0; i < longest_name - param->name.size(); ++i)
                ss << " ";
            if(param->type_flags & Control_e)
                ss << "C";
            if(param->type_flags & Input_e)
                ss << "I";
            if(param->type_flags & Output_e)
                ss << "O";

            ss << " [" << mo::Demangle::TypeToName(param->data_type) << "]\n";
            if(param->tooltip.size())
                ss << "    " << param->tooltip << "\n";
            if(param->description.size())
                ss << "    " << param->description << "\n";
            if(param->initial_value.size())
                ss << "    " << param->initial_value << "\n";
        }
    }
    auto sigs = GetSignalInfo();
    if(sigs.size())
    {
        ss << "\n----------- Signals ---------------- \n";
        int longest_name = 0;
        for(auto& slot : sigs)
            longest_name = std::max<int>(longest_name, slot->name.size());
        longest_name += 1;
        for(auto& sig : sigs)
        {
            ss << sig->name;
            for(int i = 0; i < longest_name - sig->name.size(); ++i)
                ss << " ";
            ss << " [" << sig->signature.name() << "]\n";
            if(sig->tooltip.size())
                ss << "    " << sig->tooltip << "\n";
            if(sig->description.size())
                ss << "    " << sig->description << "\n";
        }
    }
    
    auto my_slots = GetSlotInfo();
    if(my_slots .size())
    {
        ss << "\n----------- Slots ---------------- \n";
        int longest_name = 0;
        for(auto& slot : my_slots)
            longest_name = std::max<int>(longest_name, slot->name.size());
        longest_name += 1;
        for(auto& slot : my_slots)
        {
            ss << slot->name;
            for(int i = 0; i < longest_name - (int)slot->name.size(); ++i)
                ss << " ";
            ss << " [" << slot->signature.name() << "]\n";
            if(slot->tooltip.size())
                ss << "    " << slot->tooltip << "\n";
            if(slot->description.size())
                ss << "    " << slot->description << "\n";
        }
    }
    return ss.str();
}
