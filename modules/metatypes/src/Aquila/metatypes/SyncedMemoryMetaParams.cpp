#include "SyncedMemoryMetaParams.hpp"
#include "Aquila/serialization/cereal/SyncedMemory.hpp"
#include "MetaObject/params/MetaParam.hpp"
#ifdef HAVE_QT
#include "MetaObject/params/ui/Qt/OpenCV.hpp"
#include "MetaObject/params/ui/Qt/Containers.hpp"
#include "MetaObject/params/ui/Qt/TParamProxy.hpp"
#endif
#include "MetaObject/params/buffers/CircularBuffer.hpp"
#include "MetaObject/params/buffers/StreamBuffer.hpp"
#include "MetaObject/params/buffers/Map.hpp"
#include "MetaObject/params/buffers/NNStreamBuffer.hpp"
#include "MetaObject/serialization/CerealPolicy.hpp"

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
#include "MetaObject/params/detail/MetaParamImpl.hpp"
INSTANTIATE_META_PARAM(aq::SyncedMemory);
INSTANTIATE_META_PARAM(std::vector<aq::SyncedMemory>);
INSTANTIATE_META_PARAM(cv::Mat);
INSTANTIATE_META_PARAM(std::vector<cv::Mat>);

using namespace mo;
using namespace aq;
namespace mo{
    TInputParamPtr<aq::SyncedMemory>::TInputParamPtr(const std::string& name, Input_t* user_var_, Context* ctx) :
        _user_var(user_var_),
            ITInputParam<aq::SyncedMemory>(name, ctx),
            IParam(name, mo::Input_e){
    }


    bool TInputParamPtr<aq::SyncedMemory>::setInput(std::shared_ptr<IParam> param){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(ITInputParam<aq::SyncedMemory>::setInput(param)){
            if(_user_var){
                InputStorage_t data;
                if(this->_input)
                     if(this->_input->getData(data)){
                         _current_data = data;
                         *_user_var = ParamTraits<aq::SyncedMemory>::ptr(_current_data);
                         return true;
                     }
                if(this->_shared_input)
                    if(this->_shared_input->getData(data)){
                        _current_data = data;
                        *_user_var = ParamTraits<aq::SyncedMemory>::ptr(_current_data);
                        return true;
                    }
            }
            return true;
        }
        return false;
    }

    bool TInputParamPtr<aq::SyncedMemory>::setInput(IParam* param){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(ITInputParam<aq::SyncedMemory>::setInput(param)){
            if(_user_var){
                InputStorage_t data;
                if(ITInputParam<aq::SyncedMemory>::_input)
                    if(ITInputParam<aq::SyncedMemory>::_input->getData(data)){
                        _current_data = data;
                        *_user_var = ParamTraits<aq::SyncedMemory>::ptr(_current_data);
                    }
                if(ITInputParam<aq::SyncedMemory>::_shared_input)
                    if(ITInputParam<aq::SyncedMemory>::_shared_input->getData(data)){
                        _current_data = data;
                        *_user_var = ParamTraits<aq::SyncedMemory>::ptr(_current_data);
                    }
            }
            return true;
        }
        return false;
    }


    void TInputParamPtr<aq::SyncedMemory>::setUserDataPtr(Input_t* user_var_){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        _user_var = user_var_;
    }

    void TInputParamPtr<aq::SyncedMemory>::onInputUpdate(ConstStorageRef_t data, IParam* param, Context* ctx, OptionalTime_t ts, size_t fn, ICoordinateSystem* cs, UpdateFlags fg){
        if(fg == mo::BufferUpdated_e && param->checkFlags(mo::Buffer_e)){
            ITParam<aq::SyncedMemory>::_typed_update_signal(data, this, ctx, ts, fn, cs, mo::BufferUpdated_e);
            IParam::emitUpdate(ts, ctx, fn, cs, fg);
            return;
        }
        if(ctx && this->_ctx && ctx->thread_id == this->_ctx->thread_id){
            _current_data = data;
            this->_ts = ts;
            this->_fn = fn;
            if(_user_var){
                *_user_var = ParamTraits<aq::SyncedMemory>::ptr(_current_data);
            }
        }
    }

    bool TInputParamPtr<aq::SyncedMemory>::getInput(OptionalTime_t ts, size_t* fn_){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        if(_user_var){
            size_t fn;
            InputStorage_t data;
            if(ITInputParam<aq::SyncedMemory>::_shared_input){
                if(!ITInputParam<aq::SyncedMemory>::_shared_input->getData(data, ts, this->_ctx, &fn)){
                    return false;
                }
            }
            if(ITInputParam<aq::SyncedMemory>::_input)
            {
                if (!ITInputParam<aq::SyncedMemory>::_input->getData(data, ts, this->_ctx, &fn)) {
                    return false;
                }
            }
            _current_data = data;
            *_user_var = ParamTraits<aq::SyncedMemory>::ptr(_current_data);
            if (fn_)
                *fn_ = fn;
            return true;
        }
        return false;
    }


    bool TInputParamPtr<aq::SyncedMemory>::getInput(size_t fn, OptionalTime_t* ts_){
        mo::Mutex_t::scoped_lock lock(IParam::mtx());
        OptionalTime_t ts;
        if(_user_var){
            if(ITInputParam<aq::SyncedMemory>::_shared_input){
                InputStorage_t data;
                if(ITInputParam<aq::SyncedMemory>::_shared_input->getData(data, fn, this->_ctx, &ts)){
                    _current_data = data;

                    *_user_var = ParamTraits<aq::SyncedMemory>::ptr(_current_data);
                    if (ts_)
                        *ts_ = ts;
                    this->_ts = ts;
                    this->_fn = fn;
                    return true;
                }
            }
            if(ITInputParam<aq::SyncedMemory>::_input){
                InputStorage_t data;
                if(this->_input->getData(data, fn, this->_ctx, &ts)){
                    _current_data = data;
                    *_user_var = ParamTraits<aq::SyncedMemory>::ptr(_current_data);
                    if (ts_)
                        *ts_ = ts;
                    this->_ts = ts;
                    this->_fn = fn;
                    return true;
                }
            }
        }
        return false;
    }
}
