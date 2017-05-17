#include <MetaObject/params/IO/CerealPolicy.hpp>
#include <MetaObject/params/IO/CerealMemory.hpp>

#include "Aquila/Algorithm.h"
#include "Aquila/Detail/AlgorithmImpl.hpp"
#include <MetaObject/params/InputParameter.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_window.hpp>

#include <Aquila/IO/JsonArchive.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include "MetaObject/params/detail/MetaParametersDetail.hpp"
INSTANTIATE_META_PARAMETER(std::vector<rcc::shared_ptr<aq::Algorithm>>)
using namespace mo;
using namespace aq;


Algorithm::Algorithm()
{
    _pimpl = new impl();
    _enabled = true;
    _pimpl->_sync_method = SyncEvery;
}

Algorithm::~Algorithm()
{
    delete _pimpl;
    _pimpl = nullptr;
}

double Algorithm::GetAverageProcessingTime() const
{
    return 0.0;
}

void Algorithm::SetEnabled(bool value)
{
    _enabled = value;
}

bool Algorithm::IsEnabled() const
{
    return _enabled;
}
std::vector<mo::IParameter*> Algorithm::GetComponentParameters(const std::string& filter) const
{
    std::vector<mo::IParameter*> output; // = mo::IMetaObject::GetParameters(filter);
    for(auto& component: _algorithm_components)
    {
        if(component)
        {
            std::vector<mo::IParameter*> output2 = component->GetParameters(filter);
            output.insert(output.end(), output2.begin(), output2.end());
        }
    }
    return output;
}
std::vector<mo::IParameter*> Algorithm::GetAllParameters(const std::string& filter) const
{
    std::vector<mo::IParameter*> output = mo::IMetaObject::GetParameters(filter);
    for(auto& component: _algorithm_components)
    {
        if(component)
        {
            std::vector<mo::IParameter*> output2 = component->GetParameters(filter);
            output.insert(output.end(), output2.begin(), output2.end());
        }
    }
    return output;
}
bool Algorithm::Process()
{
    boost::recursive_mutex::scoped_lock lock(*_mtx);
    if(_enabled == false)
        return false;
    if(CheckInputs() == NoneValid)
    {
        return false;
    }
    if(ProcessImpl())
    {
        /*_pimpl->last_ts = _pimpl->ts;
        if(_pimpl->sync_input == nullptr && _pimpl->ts != -1)
            ++_pimpl->ts;
        if(_pimpl->_sync_method == SyncEvery && _pimpl->sync_input)
        {
            if(_pimpl->_ts_processing_queue.size() && _pimpl->ts == _pimpl->_ts_processing_queue.front())
            {
                _pimpl->_ts_processing_queue.pop();
            }
        }*/
        return true;
    }
    return false;
}
mo::IParameter* Algorithm::GetOutput(const std::string& name) const
{
    auto output = mo::IMetaObject::GetOutput(name);
    if(output)
        return output;
    if(!output)
    {
        for(auto& component: _algorithm_components)
        {
            if(component)
            {
                output = component->GetOutput(name);
                if(output)
                    return output;
            }
        }
    }
    return nullptr;
}



Algorithm::InputState Algorithm::CheckInputs()
{
    auto inputs = this->GetInputs();
    if(inputs.size() == 0)
        return AllValid;
    for(auto input : inputs)
    {
        if(!input->IsInputSet() && ! input->CheckFlags(mo::Optional_e))
        {
            LOG(trace) << "Required input (" << input->GetTreeName() << ") is not set to anything";
            return NoneValid;
        }
    }
    boost::optional<mo::time_t> ts;
    boost::optional<size_t> fn;
    // First check to see if we have a sync input, if we do then use its synchronizatio method
    // TODO: Handle processing queue
#ifdef _DEBUG
    struct InputState
    {
        InputState(const std::string& name_,
                   const boost::optional<mo::time_t>& ts_,
                   size_t fn_): name(name_), ts(ts_), fn(fn_){}
        std::string name;
        boost::optional<mo::time_t> ts;
        size_t fn;
    };

    std::vector<InputState> input_states;
#endif
    if(_pimpl->sync_input)
    {
        ts = _pimpl->sync_input->GetInputTimestamp();
        if(!ts)
            fn = _pimpl->sync_input->GetInputFrameNumber();
        if(_pimpl->_sync_method == SyncEvery)
        {

        }
    }else
    {
        // Check all inputs to see if any are timestamped.
        for(auto input : inputs)
        {
            if(input->IsInputSet())
            {
                if (input->CheckFlags(mo::Desynced_e))
                    continue;
                auto in_ts = input->GetInputTimestamp();
#ifdef _DEBUG
                input_states.emplace_back(input->GetTreeName(), in_ts, 0);
#endif
                if(!in_ts)
                    continue;
                if(in_ts)
                {
                    if(!ts)
                    {
                        ts = in_ts;
                        continue;
                    }else
                    {
                        ts = std::min<mo::time_t>(*ts, *in_ts);
                    }
                }
            }
        }
        if(!ts)
        {
            fn = -1;
            for(int i = 0; i < inputs.size(); ++i)
            {
                if(inputs[i]->IsInputSet())
                {
                    auto in_fn = inputs[i]->GetInputFrameNumber();
                    fn = std::min<size_t>(*fn, in_fn);
#ifdef _DEBUG
                    input_states.emplace_back(inputs[i]->GetTreeName(), boost::optional<mo::time_t>(), in_fn);
#endif
                }
            }
        }
    }

    // Synchronizing on timestamp

    if(ts)
    {
        size_t fn;
        for(auto input : inputs)
        {
            if(!input->GetInput(ts, &fn))
            {
                if(input->CheckFlags(mo::Desynced_e))
                    if(input->GetInput(boost::optional<mo::time_t>(), &fn))
                        continue;
                if(input->CheckFlags(mo::Optional_e))
                {
                    // If the input isn't set and it's optional then this is ok
                    if(input->GetInputParam())
                    {
                        // Input is optional and set, but couldn't get the right timestamp, error
                        LOG(debug) << "Failed to get input \"" << input->GetTreeName() << "\" at timestamp " << ts;
                    }else
                    {
                        LOG(trace) << "Optional input not set \"" << input->GetTreeName() << "\"";
                    }
                }else
                {
                    // Input is not optional
                    if (auto param = input->GetInputParam())
                    {
                        if(param->CheckFlags(mo::Unstamped_e))
                            continue;
                        LOG(trace) << "Failed to get input for \"" << input->GetTreeName() << "\" (" << param->GetTreeName() << ") at timestamp " << ts;
                        return NoneValid;
                    }else
                    {
                        LOG(trace) << "Input not set \"" << input->GetTreeName() << "\"";
                        return NoneValid;
                    }

                }
            }
        }
        if(_pimpl->ts == ts)
            return NotUpdated;
        _pimpl->ts = ts;
        _pimpl->fn = fn;
        return AllValid;
    }
    if(fn)
    {
        boost::optional<mo::time_t> ts;
        for(auto input : inputs)
        {
            if(!input->IsInputSet() && ! input->CheckFlags(Optional_e))
            {
                LOG(trace) << "Input not set \"" << input->GetTreeName() << "\"";
                return NoneValid;
            }
            if(!input->GetInput(*fn, &ts))
            {
                if(input->CheckFlags(Desynced_e))
                {
                    continue;
                }
                if (input->CheckFlags(Optional_e))
                {
                    // If the input isn't set and it's optional then this is ok
                    if (input->IsInputSet())
                    {
                        // Input is optional and set, but couldn't get the right timestamp, error
                        LOG(debug) << "Input is set to \"" << input->GetTreeName() << "\" but could not get at frame number " << *fn;
                    }
                    else
                    {
                        LOG(trace) << "Optional input not set \"" << input->GetTreeName() << "\"";
                    }
                }
                else
                {
                    // Input is not optional
                    if (auto param = input->GetInputParam())
                    {
                        LOG(trace) << "Failed to get input for \"" << input->GetTreeName() << "\" (" << param->GetTreeName() << ") at framenumber " << *fn << " actual frame number " << input->GetFrameNumber();
                        return NoneValid;
                    }
                }
            }
        }
        if(_pimpl->fn == fn)
            return NotUpdated;
        if(ts)
            _pimpl->ts = ts;
        _pimpl->fn = *fn;
        return AllValid;
    }
    return NoneValid;
}

void Algorithm::Clock(int line_number)
{

}

boost::optional<mo::time_t> Algorithm::GetTimestamp()
{
    return _pimpl->ts;
}

void Algorithm::SetSyncInput(const std::string& name)
{
    _pimpl->sync_input = GetInput(name);
    if(_pimpl->sync_input)
    {
        LOG(info) << "Updating sync parameter for " << this->GetTypeName() << " to " << name;
    }else
    {
        LOG(warning) << "Unable to set sync input for " << this->GetTypeName() << " to " << name;
    }
}
int Algorithm::SetupVariableManager(mo::IVariableManager* mgr)
{
    int count = mo::IMetaObject::SetupVariableManager(mgr);
    for(auto& child : _algorithm_components)
    {
        count += child->SetupVariableManager(mgr);
    }
    return count;
}
void Algorithm::SetSyncMethod(SyncMethod _method)
{
    if(_pimpl->_sync_method == SyncEvery && _method != SyncEvery)
    {
        //std::swap(_pimpl->_ts_processing_queue, std::queue<long long>());
        _pimpl->_ts_processing_queue = std::queue<mo::time_t>();
    }
    _pimpl->_sync_method = _method;

}
void Algorithm::onParameterUpdate(mo::Context* ctx, mo::IParameter* param)
{
    mo::IMetaObject::onParameterUpdate(ctx, param);
    if(_pimpl->_sync_method == SyncEvery)
    {
        if(param == _pimpl->sync_input)
        {
            auto ts = _pimpl->sync_input->GetInputTimestamp();
            boost::recursive_mutex::scoped_lock lock(_pimpl->_mtx);
#ifdef _MSC_VER
#ifdef _DEBUG
            /*if(_pimpl->_ts_processing_queue.size() && ts != (_pimpl->_ts_processing_queue.back() + 1))
                LOG(debug) << "Timestamp not monotonically incrementing.  Current: " << ts << " previous: " << _pimpl->_ts_processing_queue.back();
            auto itr = std::find(_pimpl->_ts_processing_queue._Get_container().begin(), _pimpl->_ts_processing_queue._Get_container().end(), ts);
            if(itr != _pimpl->_ts_processing_queue._Get_container().end())
            {
                LOG(debug) << "Timestamp (" << ts << ") exists in processing queue.";
            }*/
#endif
#endif
            if(_pimpl->sync_input->CheckFlags(mo::Buffer_e))
            {
                if(ts)
                {
                    _pimpl->_ts_processing_queue.push(*ts);
                }else
                {
                    _pimpl->_fn_processing_queue.push(_pimpl->sync_input->GetInputFrameNumber());
                }
            }

        }
    }else if (_pimpl->_sync_method == SyncNewest)
    {
        if(param == _pimpl->sync_input)
        {
            _pimpl->ts = param->GetTimestamp();
        }
    }
}
void  Algorithm::SetContext(mo::Context* ctx, bool overwrite)
{
    mo::IMetaObject::SetContext(ctx, overwrite);
    for(auto& child : _algorithm_components)
    {
        child->SetContext(ctx, overwrite);
    }
}

void Algorithm::PostSerializeInit()
{
    for(auto& child : _algorithm_components)
    {
        child->SetContext(this->_ctx);
        child->PostSerializeInit();
    }
}
void Algorithm::AddComponent(rcc::weak_ptr<Algorithm> component)
{
    _algorithm_components.push_back(component);
    mo::ISlot* slot = this->GetSlot("parameter_updated", mo::TypeInfo(typeid(void(mo::Context*, mo::IParameter*))));
    if(slot)
    {
        auto params = component->GetParameters();
        for(auto param : params)
        {
            param->RegisterUpdateNotifier(slot);
        }
    }


}
void  Algorithm::Serialize(ISimpleSerializer *pSerializer)
{
    mo::IMetaObject::Serialize(pSerializer);
    SERIALIZE(_algorithm_components);
}
