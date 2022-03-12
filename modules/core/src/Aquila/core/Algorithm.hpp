#pragma once
#include "Aquila/detail/export.hpp"
#include "IAlgorithm.hpp"
#include "ParameterSynchronizer.hpp"

#include <MetaObject/object/MetaObject.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <boost/circular_buffer.hpp>
#include <boost/fiber/recursive_mutex.hpp>
#include <boost/fiber/recursive_timed_mutex.hpp>
#include <boost/thread/recursive_mutex.hpp>

#include <queue>

#define STRINGIFY_(X) #X
#define STRINGIFY(X) STRINGIFY_(X)

#define LOG_ALGO(LEVEL, ...) this->getLogger().LEVEL(__FILE__ ":" STRINGIFY(__LINE__) " " __VA_ARGS__)

namespace aq
{
    class AQUILA_EXPORTS Algorithm : virtual public IAlgorithm
    {
      public:
        MO_DERIVE(Algorithm, IAlgorithm)

        MO_END;

        Algorithm();
        bool process() override;
        bool process(mo::IAsyncStream&) override;

        void addParam(std::shared_ptr<mo::IParam> param) override;
        void addParam(mo::IParam& param) override;

        int setupParamServer(const std::shared_ptr<mo::IParamServer>& mgr) override;
        int setupSignals(const std::shared_ptr<mo::RelayManager>& mgr) override;
        void setEnabled(bool value) override;
        bool getEnabled() const override;

        virtual mo::OptionalTime getTimestamp();

        void setSyncInput(const std::string& name) override;
        void setSyncMethod(SyncMethod method) override;
        void setSynchronizer(std::unique_ptr<ParameterSynchronizer>) override;

        void postSerializeInit() override;
        void Init(bool first_init) override;

        mo::ParamVec_t getComponentParams(const std::string& filter = "") override;
        mo::ConstParamVec_t getComponentParams(const std::string& filter = "") const override;

        mo::ConstParamVec_t getParams(const std::string& filter = "") const override;
        mo::ParamVec_t getParams(const std::string& filter = "") override;

        const mo::IControlParam* getParam(const std::string& name) const override;
        mo::IControlParam* getParam(const std::string& name) override;

        const mo::IPublisher* getOutput(const std::string& name) const override;
        mo::IPublisher* getOutput(const std::string& name) override;

        IMetaObject::ConstPublisherVec_t getOutputs(const std::string& name_filter = "") const override;
        IMetaObject::ConstPublisherVec_t getOutputs(const mo::TypeInfo& type_filter,
                                                    const std::string& name_filter = "") const override;
        IMetaObject::PublisherVec_t getOutputs(const std::string& name_filter = "") override;
        IMetaObject::PublisherVec_t getOutputs(const mo::TypeInfo& type_filter,
                                               const std::string& name_filter = "") override;

        void setStream(const mo::IAsyncStreamPtr_t& ctx) override;

        std::vector<rcc::weak_ptr<IAlgorithm>> getComponents() const override;
        void addComponent(const rcc::weak_ptr<IAlgorithm>& component) override;

        void Serialize(ISimpleSerializer* pSerializer) override;

        void setLogger(const std::shared_ptr<spdlog::logger>& logger) override;

        spdlog::logger& getLogger() const;

      protected:
        void clearModifiedInputs();
        void clearModifiedControlParams();

        InputState checkInputs() override;
        InputState syncTimestamp(const mo::Time& ts, const std::vector<mo::ISubscriber*>& inputs);
        InputState syncFrameNumber(size_t fn, const std::vector<mo::ISubscriber*>& inputs);

        bool checkModified(const std::vector<mo::ISubscriber*>& inputs) const;
        bool checkModifiedControlParams() const;

        void removeTimestampFromBuffer(const mo::Time& ts);
        void removeFrameNumberFromBuffer(size_t fn);
        mo::OptionalTime findDirectTimestamp(bool& buffered, const std::vector<mo::ISubscriber*>& inputs);
        mo::OptionalTime findBufferedTimestamp();

        void onParamUpdate(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream*) override;

        struct SyncData
        {
            SyncData(const boost::optional<mo::Time>& ts_, mo::FrameNumber fn);

            bool operator==(const SyncData& other);

            bool operator!=(const SyncData& other);

            boost::optional<mo::Time> ts;
            mo::FrameNumber fn;
        };

        bool processImpl(mo::IAsyncStream& stream) override;
        bool processImpl(mo::IDeviceStream& stream) override;
        bool processImpl() override = 0;

      private:
        std::shared_ptr<spdlog::logger> m_logger;
        mo::FrameNumber m_fn;
        mo::OptionalTime m_ts;
        mo::OptionalTime m_last_ts;
        mo::ISubscriber* m_sync_input = nullptr;
        IAlgorithm::SyncMethod m_sync_method = IAlgorithm::SyncMethod::kEVERY;
        std::queue<mo::Time> m_ts_processing_queue;
        std::queue<size_t> m_fn_processing_queue;
        mo::Mutex_t m_mtx;
        std::map<const mo::ISubscriber*, boost::circular_buffer<SyncData>> m_buffer_timing_data;
        std::vector<rcc::weak_ptr<IAlgorithm>> m_algorithm_components;
        bool m_enabled = true;
        std::unique_ptr<ParameterSynchronizer> m_synchronizer;
    };
} // namespace aq
