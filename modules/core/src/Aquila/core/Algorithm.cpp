#include <Aquila/core/Algorithm.hpp>
#include <Aquila/utilities/container.hpp>

#include <MetaObject/params/IParam.hpp>
#include <MetaObject/params/ISubscriber.hpp>
#include <MetaObject/params/buffers/IBuffer.hpp>

#include <RuntimeObjectSystem/ISimpleSerializer.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_window.hpp>

using namespace mo;
using namespace aq;

Algorithm::Algorithm()
{
    m_logger = mo::getLogger();
    m_synchronizer = std::make_unique<ParameterSynchronizer>(*m_logger);
}

void Algorithm::setEnabled(bool value)
{
    m_enabled = value;
}

bool Algorithm::getEnabled() const
{
    return m_enabled;
}

mo::ConstParamVec_t Algorithm::getComponentParams(const std::string& filter) const
{
    mo::ConstParamVec_t output; // = mo::IMetaObject::getParams(filter);
    for (auto& component : m_algorithm_components)
    {
        auto shared = component.lock();
        if (shared)
        {
            auto output2 = shared->getParams(filter);
            output.insert(output.end(), output2.begin(), output2.end());
        }
    }
    return output;
}

mo::ParamVec_t Algorithm::getComponentParams(const std::string& filter)
{
    mo::ParamVec_t output; // = mo::IMetaObject::getParams(filter);
    for (auto& component : m_algorithm_components)
    {
        auto shared = component.lock();
        if (shared)
        {
            auto output2 = shared->getParams(filter);
            output.insert(output.end(), output2.begin(), output2.end());
        }
    }
    return output;
}

mo::ParamVec_t Algorithm::getParams(const std::string& filter)
{
    mo::ParamVec_t output = mo::MetaObject::getParams(filter);
    for (auto& component : m_algorithm_components)
    {
        auto shared = component.lock();
        if (shared)
        {
            auto output2 = shared->getParams(filter);
            output.insert(output.end(), output2.begin(), output2.end());
        }
    }
    return output;
}

mo::ConstParamVec_t Algorithm::getParams(const std::string& filter) const
{
    mo::ConstParamVec_t output = mo::MetaObject::getParams(filter);
    for (auto& component : m_algorithm_components)
    {
        auto shared = component.lock();
        if (shared)
        {
            auto output2 = shared->getParams(filter);
            output.insert(output.end(), output2.begin(), output2.end());
        }
    }
    return output;
}

const mo::IControlParam* Algorithm::getParam(const std::string& name) const
{
    auto output = IAlgorithm::getParam(name);
    if (output)
    {
        return output;
    }

    for (auto& component : m_algorithm_components)
    {
        auto shared = component.lock();
        if (shared)
        {
            output = shared->getParam(name);
            if (output)
            {
                return output;
            }
        }
    }
    return output;
}

mo::IControlParam* Algorithm::getParam(const std::string& name)
{
    auto output = IAlgorithm::getParam(name);
    if (output)
    {
        return output;
    }

    for (auto& component : m_algorithm_components)
    {
        auto shared = component.lock();
        if (shared)
        {
            output = shared->getParam(name);
            if (output)
            {
                return output;
            }
        }
    }
    return output;
}

bool Algorithm::process()
{
    mo::IAsyncStream::Ptr_t stream = this->getStream();
    if (stream == nullptr)
    {
        return false;
    }
    return this->process(*stream);
}

bool Algorithm::process(mo::IAsyncStream& stream)
{
    mo::Lock_t lock(getMutex());
    if (m_enabled == false)
    {
        return false;
    }

    if (checkInputs() != InputState::kALL_VALID)
    {
        return false;
    }
    mo::IDeviceStream* dev_stream = stream.getDeviceStream();
    if (dev_stream)
    {
        if (processImpl(*dev_stream))
        {
            clearModifiedInputs();
            return true;
        }
    }
    else
    {
        if (processImpl(stream))
        {
            clearModifiedInputs();
            return true;
        }
    }

    return false;
}

bool Algorithm::processImpl(mo::IAsyncStream&)
{
    return this->processImpl();
}

bool Algorithm::processImpl(mo::IDeviceStream&)
{
    return this->processImpl();
}


void Algorithm::addParam(std::shared_ptr<mo::IParam> param)
{
    IAlgorithm::addParam(std::move(param));
}

void Algorithm::addParam(mo::IParam& param)
{
    IAlgorithm::addParam(param);

    auto inputs = this->getInputs();
    m_synchronizer->setInputs(inputs);
}

const mo::IPublisher* Algorithm::getOutput(const std::string& name) const
{
    auto output = mo::MetaObject::getOutput(name);
    if (output)
    {
        return output;
    }
    if (!output)
    {
        for (auto& component : m_algorithm_components)
        {
            auto shared = component.lock();
            if (shared)
            {
                output = shared->getOutput(name);
                if (output)
                {
                    return output;
                }
            }
        }
    }
    return nullptr;
}

mo::IPublisher* Algorithm::getOutput(const std::string& name)
{
    auto output = mo::MetaObject::getOutput(name);
    if (output)
    {
        return output;
    }
    if (!output)
    {
        for (auto& component : m_algorithm_components)
        {
            auto shared = component.lock();
            if (shared)
            {
                output = shared->getOutput(name);
                if (output)
                {
                    return output;
                }
            }
        }
    }
    return nullptr;
}

IMetaObject::PublisherVec_t Algorithm::getOutputs(const std::string& name_filter)
{
    auto outputs = mo::MetaObject::getOutputs(name_filter);
    return outputs;
}

IMetaObject::ConstPublisherVec_t Algorithm::getOutputs(const std::string& name_filter) const
{
    auto outputs = mo::MetaObject::getOutputs(name_filter);
    for (auto& component : m_algorithm_components)
    {
        auto shared = component.lock();
        if (shared)
        {
            auto comp_outputs = shared->getOutputs(name_filter);
            outputs.insert(outputs.end(), comp_outputs.begin(), comp_outputs.end());
        }
    }
    return outputs;
}

IMetaObject::PublisherVec_t Algorithm::getOutputs(const mo::TypeInfo& type_filter, const std::string& name_filter)
{
    auto outputs = mo::MetaObject::getOutputs(type_filter, name_filter);
    for (auto& component : m_algorithm_components)
    {
        auto shared = component.lock();
        if (shared)
        {
            auto comp_outputs = shared->getOutputs(type_filter, name_filter);
            outputs.insert(outputs.end(), comp_outputs.begin(), comp_outputs.end());
        }
    }
    return outputs;
}

IMetaObject::ConstPublisherVec_t Algorithm::getOutputs(const mo::TypeInfo& type_filter,
                                                       const std::string& name_filter) const
{
    auto outputs = mo::MetaObject::getOutputs(type_filter, name_filter);
    for (auto& component : m_algorithm_components)
    {
        auto shared = component.lock();
        if (shared)
        {
            auto comp_outputs = shared->getOutputs(type_filter, name_filter);
            outputs.insert(outputs.end(), comp_outputs.begin(), comp_outputs.end());
        }
    }
    return outputs;
}

bool Algorithm::checkModified(const std::vector<mo::ISubscriber*>& inputs) const
{
    for (auto input : inputs)
    {
        if (input->hasNewData())
        {
            LOG_ALGO(trace, "param {} has new data", input->getName());
            return true;
        }
    }
    return false;
}

bool Algorithm::checkModifiedControlParams() const
{
    auto params = this->getParams();
    for (auto param : params)
    {
        if (param->checkFlags(mo::ParamFlags::kCONTROL))
        {
            if (auto control = static_cast<const mo::IControlParam*>(param))
            {
                if (control->getModified())
                {
                    LOG_ALGO(trace, "control param {} has been modified", param->getName());
                    return true;
                }
            }
        }
    }
    return false;
}

Algorithm::InputState Algorithm::syncTimestamp(const mo::Time& ts, const std::vector<mo::ISubscriber*>& inputs)
{
    if (m_ts && (*m_ts == ts))
    {
        // return InputState::kNOT_UPDATED;
        if (!checkModified(inputs) && !checkModifiedControlParams())
        {
            LOG_ALGO(debug, "Timestamp already processed and no inputs have been modified");
            return InputState::kNOT_UPDATED;
        }
    }
    FrameNumber fn;
    mo::Header header(ts);
    for (auto input : inputs)
    {
        auto data = input->getData(&header);
        if (data)
        {
            LOG_ALGO(trace,
                     "Got data at {} when requesting data at {} from input {}",
                     data->getHeader().timestamp,
                     ts,
                     input->getName());
            fn = data->getHeader().frame_number;
        }
        else
        {
            if (input->checkFlags(mo::ParamFlags::kDESYNCED))
            {
                data = input->getData();
                if (data != nullptr)
                {
                    continue;
                }
            }
            if (input->checkFlags(mo::ParamFlags::kOPTIONAL))
            {
                // If the input isn't set and it's optional then this is ok
                if (auto input_param = input->getPublisher())
                {
                    // Input is optional and set, but couldn't get the right timestamp, error
                    if (auto buf_ptr = dynamic_cast<mo::buffer::IBuffer*>(input_param))
                    {
                        mo::OptionalTime start, end;
                        buf_ptr->getTimestampRange(start, end);
                        LOG_ALGO(debug,
                                 "Failed to get input '{}' at timestamp {} buffer range [{}, {}]",
                                 input->getTreeName(),
                                 ts,
                                 start,
                                 end);
                    }
                    else
                    {
                        LOG_ALGO(debug, "Failed to get input '{}' at timestamp {}", input->getTreeName(), ts);
                    }
                }
                else
                {
                    LOG_ALGO(trace, "Optional input not set '{}'", input->getTreeName());
                }
            }
            else
            {
                // Input is not optional
                if (auto param = input->getPublisher())
                {
                    if (param->checkFlags(mo::ParamFlags::kUNSTAMPED))
                    {
                        continue;
                    }
                    if (param->checkFlags(mo::ParamFlags::kBUFFER))
                    {
                        auto buffer = dynamic_cast<mo::buffer::IBuffer*>(param);
                        mo::OptionalTime begin, end;
                        if (buffer->getTimestampRange(begin, end))
                        {
                            if (begin && end)
                            {
                                LOG_ALGO(trace,
                                         "Failed to get input for '{}' ({}) at timestamp {}, input buffer: {} -> {}",
                                         input->getTreeName(),
                                         param->getTreeName(),
                                         ts,
                                         *begin,
                                         *end);
                            }
                            else
                            {
                                LOG_ALGO(trace,
                                         "Failed to get input for '{}' ({}) at timestamp {}, input buffer is empty",
                                         input->getTreeName(),
                                         param->getTreeName(),
                                         ts);
                            }
                        }
                    }
                    else
                    {
                        auto data = param->getData();
                        if (data)
                        {
                            LOG_ALGO(trace,
                                     "Failed to get input for '{}' ({}) at timestamp {}, input contains {}",
                                     input->getTreeName(),
                                     param->getTreeName(),
                                     ts,
                                     data->getHeader());
                        }
                        else
                        {
                            LOG_ALGO(trace,
                                     "Failed to get input for '{}' ({}) at timestamp {}, input is empty",
                                     input->getTreeName(),
                                     param->getTreeName(),
                                     ts);
                        }
                    }

                    return InputState::kNONE_VALID;
                }

                LOG_ALGO(trace, "Input not set '{}'", input->getTreeName());

                return InputState::kNONE_VALID;
            }
        }
    }
    m_ts = ts;
    m_fn = fn;
    LOG_ALGO(debug, "All inputs pass, ready to process");
    return Algorithm::InputState::kALL_VALID;
}

Algorithm::InputState Algorithm::syncFrameNumber(size_t fn, const std::vector<mo::ISubscriber*>& inputs)
{
    boost::optional<mo::Time> ts;
    std::vector<mo::OptionalTime> tss;
    mo::Header header(fn);
    for (auto input : inputs)
    {
        if (!input->isInputSet() && !input->checkFlags(mo::ParamFlags::kOPTIONAL))
        {
            MO_LOG(trace, "Input not set '{}'", input->getTreeName());
            return InputState::kNONE_VALID;
        }
        auto data = input->getData(&header);
        if (data)
        {
            tss.push_back(data->getHeader().timestamp);
        }
        else
        {
            if (input->checkFlags(mo::ParamFlags::kDESYNCED))
            {
                continue;
            }
            if (input->checkFlags(mo::ParamFlags::kOPTIONAL))
            {
                // If the input isn't set and it's optional then this is ok
                if (input->isInputSet())
                {
                    // Input is optional and set, but couldn't get the right timestamp, error
                    MO_LOG(
                        debug, "Input is set to \"{}\" but could not get at frame number {}", input->getTreeName(), fn);
                }
                else
                {
                    MO_LOG(trace, "Optional input not set '{}'", input->getTreeName());
                }
            }
            else
            {
                // Input is not optional
                if (auto param = input->getPublisher())
                {
                    if (param->checkFlags(mo::ParamFlags::kBUFFER))
                    {
                        auto buffer = dynamic_cast<mo::buffer::IBuffer*>(param);
                        uint64_t begin, end;
                        if (buffer->getFrameNumberRange(begin, end))
                        {
                            LOG_ALGO(trace,
                                     "Failed to get input for '{}' ({}) at framenumber {}, input buffer: {} -> {}",
                                     input->getTreeName(),
                                     param->getTreeName(),
                                     fn,
                                     begin,
                                     end);
                        }
                    }
                    else
                    {
                        auto data = param->getData();
                        if (data)
                        {
                            LOG_ALGO(trace,
                                     "Failed to get input for '{}' ({}) at timestamp {}, input contains {}",
                                     input->getTreeName(),
                                     param->getTreeName(),
                                     fn,
                                     data->getHeader());
                        }
                        else
                        {
                            LOG_ALGO(trace,
                                     "Failed to get input for '{}' ({}) at timestamp {}, input is empty",
                                     input->getTreeName(),
                                     param->getTreeName(),
                                     fn);
                        }
                    }
                    return InputState::kNONE_VALID;
                }
                else
                {
                    MO_LOG(trace, "Input not set for '{}'", input->getTreeName());
                }
            }
        }
    }
    if (m_fn == fn)
    {
        if (!checkModified(inputs))
        {
            return InputState::kNOT_UPDATED;
        }
    }
    if (m_ts)
    {
        if (std::count(tss.begin(), tss.end(), m_ts) == tss.size())
        {
            return InputState::kNOT_UPDATED;
        }
    }
    if (ts)
    {
        m_ts = ts;
    }
    m_fn = fn;
    return InputState::kALL_VALID;
}

void Algorithm::removeTimestampFromBuffer(const mo::Time& ts)
{
    mo::Lock_t lock(m_mtx);
    for (auto& itr : m_buffer_timing_data)
    {
        auto itr2 = std::find_if(itr.second.begin(), itr.second.end(), [ts](const SyncData& st) {
            if (st.ts)
            {
                return *st.ts == ts;
            }
            return false;
        });
        if (itr2 != itr.second.end())
        {
            itr.second.erase(itr2);
        }
    }
}

void Algorithm::removeFrameNumberFromBuffer(size_t fn)
{
    mo::Lock_t lock(m_mtx);
    for (auto& itr : m_buffer_timing_data)
    {
        auto itr2 = std::find_if(itr.second.begin(), itr.second.end(), [fn](const SyncData& st) {
            if (st.ts)
            {
                return st.fn == fn;
            }
            return false;
        });
        if (itr2 != itr.second.end())
        {
            itr.second.erase(itr2);
        }
    }
}

mo::OptionalTime Algorithm::findBufferedTimestamp()
{
    mo::Lock_t lock(m_mtx);
    if (!m_buffer_timing_data.empty())
    {
        // Search for the smallest timestamp common to all buffers
        std::vector<mo::Time> tss;
        for (const auto& itr : m_buffer_timing_data)
        {
            if (!itr.second.empty())
            {
                auto& ts = itr.second.front().ts;
                if (ts)
                {
                    tss.push_back(*ts);
                }
            }
        }
        // assuming time only moves forward, pick the largest of the min timestamps
        if (!tss.empty())
        {
            auto max_elem = std::max_element(tss.begin(), tss.end());
            bool all_found = true;
            // Check if the value is in the other timing buffers
            for (const auto& itr : m_buffer_timing_data)
            {
                bool found = false;
                for (const auto& itr2 : itr.second)
                {
                    if (itr2.ts && *(itr2.ts) == *max_elem)
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    all_found = false;
                    break;
                }
            }
            if (all_found)
            {
                return *max_elem;
            }
        }
    }
    return {};
}

mo::OptionalTime Algorithm::findDirectTimestamp(bool& buffered, const std::vector<ISubscriber*>& inputs)
{
    mo::OptionalTime ts;
    for (auto input : inputs)
    {
        auto input_param = input->getPublisher();
        if (input_param)
        {
            const auto buffered_flag = input_param->checkFlags(mo::ParamFlags::kBUFFER);
            const auto unstamped_flag = input_param->checkFlags(mo::ParamFlags::kUNSTAMPED);
            const auto modified_flag = input->hasNewData();
            if (!buffered_flag && !unstamped_flag)
            {
                if (modified_flag)
                {
                    auto headers = input_param->getAvailableHeaders();
                    if (!headers.empty())
                    {
                        auto in_ts = headers.back().timestamp;
                        if (in_ts)
                        {
                            if (!ts)
                            {
                                ts = in_ts;
                                continue;
                            }
                            ts = std::min<mo::Time>(*ts, *in_ts);
                        }
                    }
                }
            }
            else
            {
                buffered = true;
            }
        }
    }
    return ts;
}

Algorithm::InputState Algorithm::checkInputs()
{
    auto inputs = this->getInputs();
    if(inputs.empty())
    {
        return Algorithm::InputState::kALL_VALID;
    }
    auto next_header = this->m_synchronizer->getNextSample();

    if(next_header)
    {
        for(auto input : inputs)
        {
            auto data = input->getData(next_header.get_ptr());
            MO_ASSERT(data != nullptr);
        }
        return Algorithm::InputState::kALL_VALID;
    }
    return InputState::kNONE_VALID;
}

void Algorithm::clearModifiedInputs()
{
    auto inputs = getInputs();
    // TODO what do we do now after the refactor?
    /*for (auto input : inputs)
    {
        input->modified(false);
    }*/
}

void Algorithm::clearModifiedControlParams()
{
    auto params = this->getParams();
    for (auto param : params)
    {
        if (param->checkFlags(mo::ParamFlags::kCONTROL))
        {
            param->setModified(false);
        }
    }
}

boost::optional<mo::Time> Algorithm::getTimestamp()
{
    return m_ts;
}

void Algorithm::setSyncInput(const std::string& name)
{
    m_sync_input = getInput(name);
    if (m_sync_input)
    {
        LOG_ALGO(info, "Updating sync parameter for {} to {}", this->GetTypeName(), name);
    }
    else
    {
        LOG_ALGO(warn, "Unable to set sync input for {} to {}", this->GetTypeName(), name);
    }
}

void Algorithm::setSynchronizer(std::unique_ptr<ParameterSynchronizer> sync)
{
    m_synchronizer = std::move(sync);
}

int Algorithm::setupParamServer(const std::shared_ptr<mo::IParamServer>& mgr)
{
    int count = mo::MetaObject::setupParamServer(mgr);
    for (auto& child : m_algorithm_components)
    {
        auto shared = child.lock();
        if (shared)
        {
            count += shared->setupParamServer(mgr);
        }
    }
    return count;
}

int Algorithm::setupSignals(const std::shared_ptr<mo::RelayManager>& mgr)
{
    int cnt = IAlgorithm::setupSignals(mgr);
    for (const auto& cmp : m_algorithm_components)
    {
        auto shared = cmp.lock();
        if (shared)
        {
            cnt += shared->setupSignals(mgr);
        }
    }
    return cnt;
}

void Algorithm::setSyncMethod(SyncMethod _method)
{
    if (m_sync_method == SyncMethod::kEVERY && _method != SyncMethod::kEVERY)
    {
        // std::swap(_ts_processing_queue, std::queue<long long>());
        m_ts_processing_queue = std::queue<mo::Time>();
    }
    m_sync_method = _method;
}

void Algorithm::onParamUpdate(const mo::IParam& param, mo::Header hdr, mo::UpdateFlags fg, IAsyncStream* stream)
{
    mo::MetaObject::onParamUpdate(param, hdr, fg, stream);
    if (param.checkFlags(mo::ParamFlags::kSOURCE))
    {
        sig_update();
    }
    auto ts = hdr.timestamp;
    auto fn = hdr.frame_number;
    if (m_sync_method == SyncMethod::kEVERY)
    {
        if (&param == m_sync_input)
        {
            auto input_param = m_sync_input->getPublisher();
            mo::Lock_t lock(m_mtx);
            if (input_param && input_param->checkFlags(mo::ParamFlags::kBUFFER))
            {
                if (ts)
                {
                    if (m_ts_processing_queue.empty() || m_ts_processing_queue.back() != *ts)
                        m_ts_processing_queue.push(*ts);
                }
                else
                {
                    if (m_fn_processing_queue.empty() || m_fn_processing_queue.back() != fn)
                        m_fn_processing_queue.push(fn);
                }
            }
        }
    }
    if (fg & ct::value(UpdateFlags::kBUFFER_UPDATED))
    {
        auto in_param = dynamic_cast<const mo::ISubscriber*>(&param);
        auto buf = dynamic_cast<mo::buffer::IBuffer*>(in_param->getPublisher());

        if (in_param)
        {
            mo::Lock_t lock(m_mtx);
            auto itr = m_buffer_timing_data.find(in_param);
            if (itr == m_buffer_timing_data.end())
            {
                boost::circular_buffer<SyncData> data_buffer;
                data_buffer.set_capacity(100);
                if (buf)
                {
                    auto capacity = buf->getFrameBufferCapacity();
                    if (capacity)
                    {
                        data_buffer.set_capacity(*capacity);
                    }
                }
                auto result = m_buffer_timing_data.insert(std::make_pair(in_param, std::move(data_buffer)));
                if (result.second)
                {
                    itr = result.first;
                }
            }
            SyncData data(ts, fn);
            if (itr->second.empty() || itr->second.back() != data)
            {
                itr->second.push_back(std::move(data));
            }
        }
    }
}

void Algorithm::setStream(const mo::IAsyncStreamPtr_t& ctx)
{
    mo::MetaObject::setStream(ctx);
    for (auto& child : m_algorithm_components)
    {
        auto shared = child.lock();
        if (shared)
        {
            shared->setStream(ctx);
        }
    }
}

void Algorithm::postSerializeInit()
{
    auto stream = getStream();
    for (auto& child : m_algorithm_components)
    {
        auto shared = child.lock();
        if (shared)
        {
            shared->setStream(stream);
            shared->postSerializeInit();
        }
    }
}

void Algorithm::Init(bool first_init)
{
    if (!first_init)
    {
        for (auto& cmp : m_algorithm_components)
        {
            auto shared = cmp.lock();
            if (shared)
            {
                shared->Init(first_init);
                this->addComponent(cmp);
            }
        }
    }
    mo::MetaObject::Init(first_init);
}

void Algorithm::addComponent(const rcc::weak_ptr<IAlgorithm>& component)
{
    auto ptr = component.lock();
    if (ptr == nullptr)
    {
        return;
    }
    if (!aq::contains(m_algorithm_components, component))
    {
        m_algorithm_components.push_back(component);
    }
    mo::ISlot* slot =
        this->getSlot("param_updated", mo::TypeInfo::create<void(IParam*, Header, UpdateFlags, IAsyncStream&)>());
    if (slot)
    {
        auto params = ptr->getParams();
        for (auto param : params)
        {
            param->registerUpdateNotifier(*slot);
        }
    }
    else
    {
        m_logger->warn("Unable to get param_updated slot from self");
    }
    auto manager = getRelayManager();
    if (manager)
    {
        ptr->setupSignals(manager);
    }
    sig_componentAdded(ptr);
}

std::vector<rcc::weak_ptr<IAlgorithm>> Algorithm::getComponents() const
{
    return m_algorithm_components;
}

void Algorithm::Serialize(ISimpleSerializer* pSerializer)
{
    mo::MetaObject::Serialize(pSerializer);
    SERIALIZE(m_algorithm_components);
}

Algorithm::SyncData::SyncData(const boost::optional<mo::Time>& ts_, mo::FrameNumber fn_)
    : ts(ts_)
    , fn(fn_)
{
}

bool Algorithm::SyncData::operator==(const SyncData& other)
{
    if (ts && other.ts)
        return *ts == *other.ts;
    return fn == other.fn;
}

bool Algorithm::SyncData::operator!=(const SyncData& other)
{
    return !(*this == other);
}

void Algorithm::setLogger(const std::shared_ptr<spdlog::logger>& logger)
{
    m_logger = logger;
    m_synchronizer->setLogger(*m_logger);
}

spdlog::logger& Algorithm::getLogger() const
{
    return *m_logger;
}
