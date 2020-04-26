#pragma once
#include "Aquila/core/detail/Export.hpp"

#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>
#include <MetaObject/signals/detail/SignalMacros.hpp>

#include <ct/enum.hpp>

namespace aq
{
    class AQUILA_EXPORTS IAlgorithm : virtual public TInterface<IAlgorithm, mo::MetaObject>
    {
      public:
        MO_BEGIN(IAlgorithm)
            MO_SIGNAL(void, componentAdded, IAlgorithm*)
            MO_SIGNAL(void, update)
        MO_END;

        ENUM_BEGIN(SyncMethod, uint8_t)
            ENUM_VALUE(kEVERY, 0)  // Require every timestamp to be processed
            ENUM_VALUE(kNEWEST, 1) // process data according to the newest timestamp
        ENUM_END;

        ENUM_BEGIN(InputState, uint8_t)
            // Inputs either aren't set or aren't
            ENUM_VALUE(kNONE_VALID, 0)
            // Inputs are valid
            ENUM_VALUE(kALL_VALID, 1)
            // Inputs are valid but not updated,
            // ie we've already processed this frame
            ENUM_VALUE(kNOT_UPDATED, 2)
        ENUM_END;

        virtual bool process() = 0;

        virtual void setSyncInput(const std::string& name) = 0;
        virtual void setSyncMethod(SyncMethod method) = 0;

        virtual void setEnabled(bool value) = 0;
        virtual bool getEnabled() const = 0;

        virtual void postSerializeInit() = 0;
        virtual mo::ConstParamVec_t getComponentParams(const std::string& filter = "") const = 0;
        virtual mo::ParamVec_t getComponentParams(const std::string& filter = "") = 0;

        virtual std::vector<rcc::weak_ptr<IAlgorithm>> getComponents() const = 0;
        virtual void addComponent(const rcc::weak_ptr<IAlgorithm>& component) = 0;

        virtual void setLogger(const std::shared_ptr<spdlog::logger>& logger) = 0;

      protected:
        virtual InputState checkInputs() = 0;
        virtual bool processImpl() = 0;
    };

} // namespace aq
