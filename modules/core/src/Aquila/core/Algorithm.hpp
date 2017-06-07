#pragma once
#include "Aquila/core/detail/Export.hpp"
#include <MetaObject/object/MetaObject.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>

namespace aq{
    class AQUILA_EXPORTS Algorithm :
            public TInterface<Algorithm, mo::IMetaObject>{
    public:
        enum SyncMethod{
            SyncEvery = 0, // Require every timestamp to be processed
            SyncNewest     // process data according to the newest timestamp
        };
        enum InputState{
            // Inputs either aren't set or aren't
            NoneValid = 0,
            // Inputs are valid
            AllValid = 1,
            // Inputs are valid but not updated,
            // ie we've already processed this frame
            NotUpdated = 2
        };
        Algorithm();
        virtual ~Algorithm();

        virtual bool       process();

        virtual int        setupVariableManager(mo::IVariableManager* mgr);
        virtual void       setEnabled(bool value);
        bool               getEnabled() const;

        virtual boost::optional<mo::Time_t> getTimestamp();

        void               setSyncInput(const std::string& name);
        void               setSyncMethod(SyncMethod method);
        virtual void       postSerializeInit();

        std::vector<mo::IParam*> getComponentParams(const std::string& filter = "") const;
        std::vector<mo::IParam*> getAllParams(const std::string& filter = "") const;
        mo::IParam* getOutput(const std::string& name) const;
        template<class T>
        mo::ITParam<T>* getOutput(const std::string& name) const
        {
            return mo::IMetaObject::getOutput<T>(name);
        }
        void  setContext(const mo::ContextPtr_t& ctx, bool overwrite = false);
        const std::vector<rcc::weak_ptr<Algorithm>>& getComponents() const
        {
            return _algorithm_components;
        }
        void  Serialize(ISimpleSerializer *pSerializer);
        virtual void addComponent(rcc::weak_ptr<Algorithm> component);
    protected:
        virtual InputState checkInputs();
        virtual bool processImpl() = 0;

        virtual void onParamUpdate(mo::IParam*, mo::Context*, mo::OptionalTime_t, size_t, mo::ICoordinateSystem*, mo::UpdateFlags);
        bool _enabled;
        struct impl;
        impl* _pimpl;
        std::vector<rcc::weak_ptr<Algorithm>> _algorithm_components;
    };
}
