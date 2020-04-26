#pragma once
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include <MetaObject/signals/detail/SignalMacros.hpp>
#include <MetaObject/signals/detail/SlotMacros.hpp>
#include <Aquila/core/detail/Export.hpp>

namespace aq
{
    namespace gui
    {
        // This class emits void(void) signals based on keyboard key presses
        class AQUILA_EXPORTS KeyboardSignalController: public mo::MetaObject
        {
        public:
            using SignalMap_t = std::map<int, std::string>;
            MO_BEGIN(KeyboardSignalController)
                MO_SLOT(void, on_key, int)
                PARAM(bool, print_unused_keys, false)
                PARAM(SignalMap_t, signal_map, {})
                PARAM_UPDATE_SLOT(signal_map)
            MO_END;
        private:
            std::map<std::string, std::unique_ptr<mo::TSignal<void(void)>>> m_signals;
            std::map<int, mo::TSignal<void(void)>*> m_key_map;

        };
    }
}
