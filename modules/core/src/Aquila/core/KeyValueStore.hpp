#pragma once
#include <Aquila/detail/export.hpp>

#include "MetaObject/logging/logging.hpp"
#include "MetaObject/params/IParam.hpp"
#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/core/TypeTable.hpp>

#include <MetaObject/params/TControlParam.hpp>

#include <memory>

struct SystemTable;

namespace aq
{
    class AQUILA_EXPORTS KeyValueStore
    {
      public:
        static std::shared_ptr<KeyValueStore> instance();
        static std::shared_ptr<KeyValueStore> instance(SystemTable*);

        KeyValueStore(const KeyValueStore&) = delete;
        KeyValueStore& operator=(const KeyValueStore&) = delete;
        KeyValueStore() = default;
        KeyValueStore(KeyValueStore&&) = delete;
        KeyValueStore& operator=(KeyValueStore&&) = delete;

        virtual ~KeyValueStore();
        virtual const mo::IControlParam* getValue(const std::string& name) = 0;
        virtual mo::IControlParam* getMutableValue(const std::string& name) = 0;
        virtual void setValue(const std::string& name, std::unique_ptr<mo::IControlParam>&& para) = 0;
        virtual std::vector<std::string> listKeys() const = 0;

        template <class T>
        T read(const std::string& name);
        template <class T>
        T read(const std::string& name, const T& default_value);

        template <class T>
        void write(const std::string& name, T&& value);

        void parseArgs(std::vector<std::string>&& argv);
    };

    /////////////////////////////////////////////////////////////////////////////
    ///       IMPLEMENTATION
    /////////////////////////////////////////////////////////////////////////////

    template <class T>
    T KeyValueStore::read(const std::string& name)
    {
        const auto param = getValue(name);
        MO_ASSERT(param);
        MO_ASSERT(param->getTypeInfo() == mo::TypeInfo::create<T>());
        auto tparam = dynamic_cast<const mo::ITControlParam<T>*>(param);
        MO_ASSERT(tparam);
        return tparam->getValue();
    }

    template <class T>
    T KeyValueStore::read(const std::string& name, const T& default_value)
    {
        const auto param = getValue(name);
        if (param == nullptr)
        {
            MO_LOG(debug, "No key with name: {} available keys: {}", name, listKeys());
            return default_value;
        }
        if (param->getTypeInfo() != mo::TypeInfo::create<T>())
        {
            MO_LOG(debug,
                   "existing value for {} of type {} does not match requested type {}",
                   name,
                   mo::TypeTable::instance()->typeToName(param->getTypeInfo()),
                   mo::TypeTable::instance()->typeToName(mo::TypeInfo::create<T>()));
            return default_value;
        }
        auto tparam = dynamic_cast<const mo::ITControlParam<T>*>(param);
        if (tparam == nullptr)
        {
            return default_value;
        }
        return tparam->getValue();
    }

    template <class T>
    void KeyValueStore::write(const std::string& name, T&& value)
    {
        auto param = getMutableValue(name);
        if (param == nullptr)
        {
            std::unique_ptr<mo::TControlParam<ct::decay_t<T>>> ptr(new mo::TControlParam<ct::decay_t<T>>());
            ptr->setName(name);
            ptr->setValue(std::move(value));
            ptr->setFlags(mo::ParamFlags::kSTATE);
            setValue(name, std::move(ptr));
        }
        else
        {
            auto tparam = dynamic_cast<mo::ITControlParam<ct::decay_t<T>>*>(param);
            MO_ASSERT(tparam);
            tparam->setValue(std::move(value));
        }
    }
    std::shared_ptr<KeyValueStore> KeyValueStore::instance()
    {
        return mo::singleton<KeyValueStore>();
    }
} // namespace aq
