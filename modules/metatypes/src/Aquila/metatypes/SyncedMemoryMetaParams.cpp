#include "SyncedMemoryMetaParams.hpp"
#include "Aquila/serialization/cereal/SyncedMemory.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include "MetaObject/params/UI/Qt/OpenCV.hpp"
#include "MetaObject/params/UI/Qt/Containers.hpp"
#include "MetaObject/params/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/params/buffers/CircularBuffer.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/params/buffers/Map.hpp"
#include "MetaObject/params/buffers/NNStreamBuffer.hpp"
#include "MetaObject/params/IO/CerealPolicy.hpp"

#ifdef MO_EXPORTS
#undef MO_EXPORTS
#endif
#if (defined WIN32 || defined _WIN32 || defined WINCE || defined __CYGWIN__)
#  define MO_EXPORTS __declspec(dllexport)
#elif defined __GNUC__ && __GNUC__ >= 4
#  define MO_EXPORTS __attribute__ ((visibility ("default")))
#else
#  define MO_EXPORTS
#endif
#include "MetaObject/params/detail/MetaParametersDetail.hpp"
INSTANTIATE_META_PARAM(aq::SyncedMemory);
INSTANTIATE_META_PARAM(std::vector<aq::SyncedMemory>);
INSTANTIATE_META_PARAM(cv::Mat);
INSTANTIATE_META_PARAM(std::vector<cv::Mat>);

using namespace mo;
using namespace aq;
TypedInputParameterPtr<SyncedMemory>::TypedInputParameterPtr(const std::string& name,
                                                             const SyncedMemory** userVar_, Context* ctx) :
        userVar(userVar_),
        ITypedInputParameter<SyncedMemory>(name, ctx),
        IParameter(name, Input_e, {}, ctx),
        ITParam<SyncedMemory>(name, Input_e, {}, ctx)
{
}

bool TypedInputParameterPtr<SyncedMemory>::SetInput(std::shared_ptr<IParameter> param)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    if(ITypedInputParameter<SyncedMemory>::SetInput(param))
    {
        if(userVar)
        {
            if(this->input)
                *userVar = this->input->GetDataPtr();
            if(this->shared_input)
                *userVar = this->shared_input->GetDataPtr();
        }
        return true;
    }
    return false;
}

bool TypedInputParameterPtr<SyncedMemory>::SetInput(IParameter* param)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    if(ITypedInputParameter<SyncedMemory>::SetInput(param))
    {
        if(userVar)
        {
            if(this->input)
                *userVar = this->input->GetDataPtr();
            if(this->shared_input)
                *userVar = this->shared_input->GetDataPtr();
        }
        return true;
    }
    return false;
}

void TypedInputParameterPtr<SyncedMemory>::SetUserDataPtr(const SyncedMemory** user_var_)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    userVar = user_var_;
}

void TypedInputParameterPtr<SyncedMemory>::onInputUpdate(Context* ctx, IParameter* param)
{
    (void)param;
    if(this->input)
    {
        boost::recursive_mutex::scoped_lock lock(this->input->mtx());
        //this->Commit(this->input->GetTimestamp(), ctx, this->input->GetFrameNumber(), this->input->GetCoordinateSystem());
        this->_update_signal(ctx, this);
        if((ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id) || (ctx == nullptr &&  this->_ctx == nullptr))
        {
            if(userVar)
                *userVar = this->input->GetDataPtr();
        }
    }else if(this->shared_input)
    {
        boost::recursive_mutex::scoped_lock lock(this->shared_input->mtx());
        //this->Commit(this->shared_input->GetTimestamp(), ctx, this->shared_input->GetFrameNumber(), this->shared_input->GetCoordinateSystem());
        this->_update_signal(ctx, this);
        if((ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id) || ((ctx == nullptr) &&  (this->_ctx == nullptr)))
        {
            if(userVar)
                *userVar = this->shared_input->GetDataPtr();
        }
    }
}


bool TypedInputParameterPtr<SyncedMemory>::getInput(boost::optional<mo::Time_t> ts, size_t* fn_)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    if(userVar)
    {
        if(this->shared_input)
        {
            size_t fn;
            auto ptr = this->shared_input->GetDataPtr(ts, this->_ctx, &fn);
            if(ptr)
            {
                this->current = *ptr;
                *userVar = &current;
                if (!current.empty())
                {
                    this->_ts = ts;
                    if (fn_)
                        *fn_ = fn;
                    this->_fn = fn;
                    return true;
                }
            }
        }
        if(this->input)
        {
            size_t fn;
            auto ptr = this->input->GetDataPtr(ts, this->_ctx, &fn);
            if(ptr)
            {
                this->current = *ptr;
                *userVar = &current;
                if (!current.empty())
                {
                    this->_ts = ts;
                    if (fn_)
                        *fn_ = fn;
                    this->_fn = fn;
                    return true;
                }
            }
        }
    }
    return false;
}

bool TypedInputParameterPtr<SyncedMemory>::getInput(size_t fn, boost::optional<mo::Time_t>* ts_)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    boost::optional<mo::Time_t> ts;
    if(userVar)
    {
        if(this->shared_input)
        {
            *userVar = this->shared_input->GetDataPtr(fn, this->_ctx, &ts);
            if(*userVar != nullptr && !(*userVar)->empty())
            {
                if(ts_)
                    *ts_ = ts;
                this->_ts = ts;
                this->_fn = fn;
                return true;
            }
        }
        if(this->input)
        {
            *userVar = this->input->GetDataPtr(fn, this->_ctx, &ts);
            if(*userVar != nullptr && !(*userVar)->empty())
            {
                if(ts_)
                    *ts_ = ts;
                this->_ts = ts;
                this->_fn = fn;
                return true;
            }
        }
    }
    return false;
}

void TypedInputParameterPtr<SyncedMemory>::onInputDelete(IParameter const* param)
{
    boost::recursive_mutex::scoped_lock lock(IParameter::mtx());
    if(param == this->input || param == this->shared_input.get())
    {
        this->shared_input.reset();
        this->input = nullptr;
    }
}
