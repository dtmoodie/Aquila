#include "MetaObject/object/MetaObjectInfo.hpp"
#include "ViewControllers.hpp"
#include <MetaObject/runtime_reflection/visitor_traits/map.hpp>
#include <MetaObject/types/cereal_map.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>
#include <iostream>
namespace aq {
namespace gui {
void KeyboardSignalController::on_key(int key) {
  auto itr = m_key_map.find(key);
  if (itr != m_key_map.end()) {
    (*itr->second)();
  } else {
    if (print_unused_keys) {
      std::cout << "Received key " << key << std::endl;
    }
  }
}


void KeyboardSignalController::on_signal_map_modified(mo::IParam *, mo::Header,
                                                      mo::UpdateFlags) {
  for (const auto &kvp : signal_map) {
    auto itr = m_signals.find(kvp.second);
    if (itr == m_signals.end()) {
      std::unique_ptr<mo::TSignal<void(void)>> sig(
          new mo::TSignal<void(void)>());
      m_key_map[kvp.first] = sig.get();
      addSignal(sig.get(), kvp.second);
      m_signals[kvp.second] = std::move(sig);
    }
  }
}
}
}

using namespace aq::gui;
MO_REGISTER_CLASS(KeyboardSignalController)
