#pragma once
#include "IDataStream.hpp"

#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/detail/MetaObjectMacrosImpl.hpp>
#include <MetaObject/thread/ThreadHandle.hpp>
#include <boost/thread.hpp>
#include <opencv2/core/cuda.hpp>

#define DS_END_(N)                                              \
    SIGNAL_INFO_END(N)                                          \
    SLOT_INFO_END(N)                                            \
    PARAM_INFO_END(N)                                           \
    SIGNALS_END(N)                                              \
    SLOT_END(N)                                                 \
    void initParams(bool firstInit)                             \
    {                                                           \
        _init_params(firstInit, mo::_counter_<N - 1>());        \
        _init_parent_params(firstInit);                         \
    }                                                           \
    void serializeParams(ISimpleSerializer* pSerializer)        \
    {                                                           \
        _serialize_params(pSerializer, mo::_counter_<N - 1>()); \
        _serialize_parent_params(pSerializer);                  \
    }                                                           \
    void             initOutputs() {}                           \
    static const int _DS_N_ = N

namespace aq {
class AQUILA_EXPORTS DataStream : public IDataStream {
public:
    DataStream();
    virtual ~DataStream();
    MO_BEGIN(DataStream)
    MO_SIGNAL(void, StartThreads)
    MO_SIGNAL(void, StopThreads)

    MO_SLOT(void, startThread)
    MO_SLOT(void, input_changed, nodes::Node*, mo::InputParam*)
    MO_SLOT(void, stopThread)
    MO_SLOT(void, pauseThread)
    MO_SLOT(void, resumeThread)
    MO_SLOT(void, node_updated, nodes::Node*)
    MO_SLOT(void, update)
    MO_SLOT(void, param_updated, mo::IMetaObject*, mo::IParam*)
    MO_SLOT(void, param_added, mo::IMetaObject*, mo::IParam*)
    MO_SLOT(void, run_continuously, bool)
    MO_SLOT(int, process)
    DS_END_(__COUNTER__);

    std::vector<rcc::weak_ptr<aq::nodes::Node> > getTopLevelNodes();
    virtual mo::ContextPtr_t                     getContext();
    virtual void initCustom(bool firstInit);
    virtual std::shared_ptr<mo::IVariableManager>      getVariableManager();
    virtual mo::RelayManager*                          getRelayManager();
    virtual IParameterBuffer*                          getParameterBuffer();
    virtual rcc::weak_ptr<WindowCallbackHandler>       getWindowCallbackManager();
    virtual std::vector<rcc::shared_ptr<nodes::Node> > getNodes() const;
    virtual std::vector<rcc::shared_ptr<nodes::Node> > getAllNodes() const;
    virtual bool loadDocument(const std::string& document, const std::string& prefered_loader = "");
    virtual std::vector<rcc::shared_ptr<nodes::Node> > addNode(const std::string& nodeName);
    virtual void addNode(rcc::shared_ptr<nodes::Node> node);
    virtual void addNodeNoInit(rcc::shared_ptr<nodes::Node> node);
    virtual void addNodes(std::vector<rcc::shared_ptr<nodes::Node> > node);
    virtual void removeNode(rcc::shared_ptr<nodes::Node> node);
    virtual void removeNode(nodes::Node* node);
    virtual nodes::Node* getNode(const std::string& nodeName);
    virtual bool saveStream(const std::string& filename);
    virtual bool loadStream(const std::string& filename);
    template <class T>
    void load(T& ar);
    template <class T>
    void save(T& ar) const;

    void addVariableSink(IVariableSink* sink);
    void removeVariableSink(IVariableSink* sink);

protected:
    friend class IDataStream;
    virtual void addChildNode(rcc::shared_ptr<nodes::Node> node);
    virtual void removeChildNode(rcc::shared_ptr<nodes::Node> node);
    virtual std::unique_ptr<ISingleton>& getSingleton(mo::TypeInfo type);
    virtual std::unique_ptr<ISingleton>& getIObjectSingleton(mo::TypeInfo type);

    std::map<mo::TypeInfo, std::unique_ptr<ISingleton> > _singletons;
    std::map<mo::TypeInfo, std::unique_ptr<ISingleton> > _iobject_singletons;
    int                                   stream_id;
    size_t                                _thread_id;
    std::shared_ptr<mo::IVariableManager> variable_manager;
    std::shared_ptr<mo::RelayManager>     relay_manager;
    std::shared_ptr<IParameterBuffer>     _parameter_buffer;
    std::mutex                            nodes_mtx;
    mo::ThreadHandle                      _processing_thread;
    volatile bool                         dirty_flag;
    std::vector<IVariableSink*>           variable_sinks;
    // These are threads for attempted connections
    std::vector<boost::thread*>                connection_threads;
    std::vector<rcc::shared_ptr<nodes::Node> > top_level_nodes;
    std::vector<rcc::weak_ptr<nodes::Node> >   child_nodes;
    rcc::shared_ptr<WindowCallbackHandler>     _window_callback_handler;
};
}
