#include "KeyValueStore.hpp"
#include <MetaObject/core/SystemTable.hpp>

#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#include <ct/reflect/print-container-inl.hpp>

#include <map>
#include <unordered_map>
namespace aq
{
    class KeyValueStoreImpl : public KeyValueStore
    {
      public:
        const mo::IControlParam* getValue(const std::string& name) override
        {
            auto itr = m_storage.find(name);
            if (itr != m_storage.end())
                return itr->second.get();
            return nullptr;
        }

        mo::IControlParam* getMutableValue(const std::string& name) override
        {
            auto itr = m_storage.find(name);
            if (itr != m_storage.end())
                return itr->second.get();
            return nullptr;
        }

        void setValue(const std::string& name, std::unique_ptr<mo::IControlParam>&& value) override
        {
            m_storage[name] = std::move(value);
        }

        std::vector<std::string> listKeys() const override
        {
            std::vector<std::string> output;
            for (const auto& kvp : m_storage)
                output.push_back(kvp.first);

            return output;
        }

      private:
        std::map<std::string, std::unique_ptr<mo::IControlParam>> m_storage;
    };

    KeyValueStore::~KeyValueStore()
    {
    }

    std::shared_ptr<KeyValueStore> KeyValueStore::instance(SystemTable* table)
    {
        return table->getSingleton<KeyValueStore, KeyValueStoreImpl>();
    }

    void KeyValueStore::parseArgs(std::vector<std::string>&& argv)
    {

        for (const auto& kvp : argv)
        {
            auto pos = kvp.find('=');
            if (pos != std::string::npos)
            {
                auto k = kvp.substr(0, pos);
                auto v = kvp.substr(pos + 1);
                if (v[0] == '$')
                {
                    auto sub = v.substr(1);
                    auto val = std::getenv(sub.c_str());
                    if (val)
                    {
                        v = val;
                    }
                }
                MO_LOG(info, "parsed {}={}", k, v);
                this->write(k, v);
            }
        }
        this->write("argv", std::move(argv));
    }
} // namespace aq
