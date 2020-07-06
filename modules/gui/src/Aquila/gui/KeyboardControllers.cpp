#include "MetaObject/object/MetaObjectInfo.hpp"
#include "ViewControllers.hpp"

#include <MetaObject/runtime_reflection/visitor_traits/map.hpp>
#include <MetaObject/types/cereal_map.hpp>

#include <ct/reflect/print-container-inl.hpp>

#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>

#include <iostream>
namespace aq
{
    namespace gui
    {
        void KeyboardSignalController::on_key(int key)
        {
            auto itr = m_key_map.find(key);
            if (itr != m_key_map.end())
            {
                (*itr->second)();
            }
            else
            {
                if (print_unused_keys)
                {
                    std::cout << "Received key " << key << std::endl;
                }
            }
        }

        void KeyboardSignalController::on_signal_map_modified(const mo::IParam&,
                                                              mo::Header,
                                                              mo::UpdateFlags,
                                                              mo::IAsyncStream& stream)
        {
            for (const auto& kvp : signal_map)
            {
                auto itr = m_signals.find(kvp.second);
                if (itr == m_signals.end())
                {
                    std::unique_ptr<mo::TSignal<void(void)>> sig(new mo::TSignal<void(void)>());
                    m_key_map[kvp.first] = sig.get();
                    addSignal(*sig, kvp.second);
                    m_signals[kvp.second] = std::move(sig);
                }
            }
        }
    } // namespace gui
} // namespace aq

namespace std
{
    ostream& operator<<(ostream& os, const std::map<int, std::string> obj)
    {

        if (!obj.empty())
        {
            os << "{";
            size_t count = 0;

            for (const auto& itr : obj)
            {
                if (count != 0)
                {
                    os << " ";
                }

                os << itr.first << ":";
                os << itr.second;

                ++count;
            }
            os << "}";
        }
        return os;
    }
} // namespace std

using namespace aq::gui;
MO_REGISTER_CLASS(KeyboardSignalController)
