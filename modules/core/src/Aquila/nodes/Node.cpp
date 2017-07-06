#include "Aquila/nodes/Node.hpp"
#include "Aquila/core/IDataStream.hpp"
#include "Aquila/nodes/NodeFactory.hpp"
#include "Aquila/nodes/NodeInfo.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include <Aquila/rcc/SystemTable.hpp>
#include <Aquila/rcc/external_includes/cv_videoio.hpp>
//#include <Aquila/utilities/cuda/GpuMatAllocators.h>
#include "Aquila/core/detail/AlgorithmImpl.hpp"
//#include <Aquila/serialization/cereal/memory.hpp>

#include "RuntimeObjectSystem/ISimpleSerializer.h"
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/object/MetaObject.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/bind.hpp>
#include <boost/date_time.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/log/trivial.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/thread.hpp>
#include <boost/thread.hpp>

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <thrust/system/system_error.h>

#include <future>
#include <regex>

using namespace aq;
using namespace aq::nodes;
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

#define EXCEPTION_TRY_COUNT 10

#define CATCH_MACRO                                                            \
    catch (mo::ExceptionWithCallStack<cv::Exception> & e) {                    \
        LOG_NODE(error) << e.what() << "\n"                                    \
                        << e.callStack();                                      \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (thrust::system_error & e) {                                         \
        LOG_NODE(error) << e.what();                                           \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (mo::ExceptionWithCallStack<std::string> & e) {                      \
        LOG_NODE(error) << std::string(e) << "\n"                              \
                        << e.callStack();                                      \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (mo::IExceptionWithCallStackBase & e) {                              \
        LOG_NODE(error) << "Exception thrown with callstack: \n"               \
                        << e.callStack();                                      \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (boost::thread_resource_error & err) {                               \
        LOG_NODE(error) << err.what();                                         \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (boost::thread_interrupted & err) {                                  \
        LOG_NODE(error) << "Thread interrupted";                               \
        /* Needs to pass this back up to the chain to the processing thread.*/ \
        /* That way it knowns it needs to exit this thread */                  \
        throw err;                                                             \
    }                                                                          \
    catch (boost::thread_exception & err) {                                    \
        LOG_NODE(error) << err.what();                                         \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (cv::Exception & err) {                                              \
        LOG_NODE(error) << err.what();                                         \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (boost::exception & err) {                                           \
        LOG_NODE(error) << "Boost error";                                      \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (std::exception & err) {                                             \
        LOG_NODE(error) << err.what();                                         \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }                                                                          \
    catch (...) {                                                              \
        LOG_NODE(error) << "Unknown exception";                                \
        ++_pimpl_node->throw_count;                                            \
        if (_pimpl_node->throw_count > EXCEPTION_TRY_COUNT)                    \
            _pimpl_node->disable_due_to_errors = true;                         \
    }

std::string NodeInfo::Print(IObjectInfo::Verbosity verbosity) const {
    return mo::IMetaObjectInfo::Print(verbosity);
}

namespace aq {
namespace nodes {
    class NodeImpl {
    public:
        long long   throw_count           = 0;
        bool        disable_due_to_errors = false;
        std::string tree_name;
        long long   iterations_since_execution    = 0;
        const char* last_execution_failure_reason = 0;
#ifdef _DEBUG
        std::vector<long long> timestamps;
#endif
    };
}
}

std::vector<std::string> Node::listConstructableNodes(const std::string& filter) {
    auto                     constructors = mo::MetaObjectFactory::instance()->getConstructors(s_interfaceID);
    std::vector<std::string> output;
    for (IObjectConstructor* constructor : constructors) {
        if (filter.size()) {
            if (std::string(constructor->GetName()).find(filter) != std::string::npos) {
                output.emplace_back(constructor->GetName());
            }
        } else {
            output.emplace_back(constructor->GetName());
        }
    }
    return output;
}

Node::Node() {
    _modified = true;
    _pimpl_node.reset(new NodeImpl());
}

bool Node::connectInput(rcc::shared_ptr<Node> node, const std::string& output_name, const std::string& input_name, mo::ParamType type) {
    auto output = node->getOutput(output_name);
    auto input  = this->getInput(input_name);
    if (output && input) {
        if (this->IMetaObject::connectInput(input, node.get(), output, type)) {
            addParent(node.get());
            return true;
        } else {
            return false;
        }
    }
    if (output == nullptr) {
        auto outputs = node->getOutputs();
        auto f       = [&outputs]() -> std::string {
            std::stringstream ss;
            for (auto& output : outputs) {
                ss << output->getName() << " ";
            }
            return ss.str();
        };
        MO_LOG(debug) << "Unable to find output with name \"" << output_name << "\" in node \"" << node->getTreeName() << "\".  Existing outputs: " << f();
        ;
    }
    if (input == nullptr) {
        auto outputs = node->getInputs();
        auto f       = [&outputs]() -> std::string {
            std::stringstream ss;
            for (auto& output : outputs) {
                ss << output->getName() << " ";
            }
            return ss.str();
        };
        MO_LOG(debug) << "Unable to find input with name \"" << input_name << "\" in node \"" << this->getTreeName() << "\". Existing inputs: " << f();
    }
    return false;
}
bool Node::connectInput(rcc::shared_ptr<Node> output_node, mo::IParam* output_param, mo::InputParam* input_param, mo::ParamType type) {
    if (this->IMetaObject::connectInput(input_param, output_node.get(), output_param, type)) {
        addParent(output_node.get());
        Node* This = this;
        sig_input_changed(This, input_param);
        return true;
    } else {
        return false;
    }
}

Algorithm::InputState Node::checkInputs() {
    if (_pimpl->_sync_method == Algorithm::SyncEvery && _pimpl->_ts_processing_queue.size() != 0)
        _modified = true;
    /*if(_modified == false)
    {
        MO_LOG_EVERY_N(trace, 10) << "_modified == false for " << getTreeName();
        return Algorithm::NoneValid;
    }*/

    return Algorithm::checkInputs();
}

void Node::onParamUpdate(mo::IParam* param, mo::Context* ctx, mo::OptionalTime_t ts, size_t fn, mo::ICoordinateSystem* cs, mo::UpdateFlags fg) {
    Algorithm::onParamUpdate(param, ctx, ts, fn, cs, fg);
    if (param->checkFlags(mo::Control_e)) {
        _modified = true;
    }
    if (_pimpl_node->disable_due_to_errors && param->checkFlags(mo::Control_e)) {
        _pimpl_node->throw_count--;
        _pimpl_node->disable_due_to_errors = false;
    }
}

bool Node::process() {
    ++_pimpl_node->iterations_since_execution;
    if (_pimpl_node->iterations_since_execution % 100 == 0) {
        MO_LOG(debug) << this->getTreeName() << " has not executed in " << _pimpl_node->iterations_since_execution << " iterations due to "
                      << (_pimpl_node->last_execution_failure_reason ? _pimpl_node->last_execution_failure_reason : "");
    }
    if (_enabled == true && _pimpl_node->disable_due_to_errors == false) {
        mo::scoped_profile       profiler(this->getTreeName().c_str(), nullptr, nullptr, cudaStream());
        mo::Mutex_t::scoped_lock lock(*_mtx);
        {
            auto input_state = checkInputs();
            if (input_state == Algorithm::NoneValid) {
                _pimpl_node->last_execution_failure_reason = "No valid inputs";
                return false;
            }
            if (input_state == Algorithm::NotUpdated && _modified == false) {
                _pimpl_node->last_execution_failure_reason = "Inputs not updated and parameters not updated";
                return false;
            }
        }
        _modified = false;

        try {
            _pimpl_node->last_execution_failure_reason = "Exception thrown";
            if (!processImpl()) {
                _pimpl_node->iterations_since_execution = 0;
            }
        }
        CATCH_MACRO
        _pimpl_node->last_execution_failure_reason = 0;
        _pimpl_node->iterations_since_execution    = 0;
    }

    for (rcc::shared_ptr<Node>& child : _children) {
        if (child) {
            if (child && child->_ctx.get() && this->_ctx.get()) {
                if (child->_ctx.get()->thread_id == this->_ctx.get()->thread_id) {
                    child->process();
                }
            } else {
                child->process();
            }
        }
    }
    return true;
}

void Node::reset() {
    Init(false);
}

Node::Ptr Node::addChild(Node* child) {
    return addChild(Node::Ptr(child));
}

Node::Ptr Node::addChild(Node::Ptr child) {
    if (_ctx.get() && mo::getThisThread() != _ctx.get()->thread_id) {
        std::future<Ptr>  result;
        std::promise<Ptr> promise;
        mo::ThreadSpecificQueue::push(
            std::bind(
                [this, &promise, child]() {
                    promise.set_value(this->addChild(child));
                }),
            _ctx.get()->thread_id, this);
        result = promise.get_future();
        result.wait();
        return result.get();
    }
    if (child == nullptr)
        return child;
    if (std::find(_children.begin(), _children.end(), child) != _children.end())
        return child;
    if (child == this) // This can happen based on a bad user config
        return child;
    int count = 0;
    for (size_t i = 0; i < _children.size(); ++i) {
        if (_children[i] && _children[i]->GetTypeName() == child->GetTypeName())
            ++count;
    }
    _children.push_back(child);
    child->setDataStream(getDataStream());
    child->addParent(this);
    child->setContext(this->_ctx, false);
    std::string node_name = child->GetTypeName();
    child->setUniqueId(count);
    child->setParamRoot(child->getTreeName());
    MO_LOG(trace) << "[ " << getTreeName() << " ]"
                  << " Adding child " << child->getTreeName();
    return child;
}

Node::Ptr Node::getChild(const std::string& treeName) {
    mo::Mutex_t::scoped_lock lock(*_mtx);
    for (size_t i = 0; i < _children.size(); ++i) {
        if (_children[i]->getTreeName() == treeName)
            return _children[i];
    }
    for (size_t i = 0; i < _children.size(); ++i) {
        if (_children[i]->getTreeName() == treeName)
            return _children[i];
    }
    return Node::Ptr();
}

Node::Ptr Node::getChild(const int& index) {
    return _children[index];
}
std::vector<Node::Ptr> Node::getChildren() {
    return _children;
}

void Node::swapChildren(int idx1, int idx2) {

    std::iter_swap(_children.begin() + idx1, _children.begin() + idx2);
}

void Node::swapChildren(const std::string& name1, const std::string& name2) {

    auto itr1 = _children.begin();
    auto itr2 = _children.begin();
    for (; itr1 != _children.begin(); ++itr1) {
        if ((*itr1)->getTreeName() == name1)
            break;
    }
    for (; itr2 != _children.begin(); ++itr2) {
        if ((*itr2)->getTreeName() == name2)
            break;
    }
    if (itr1 != _children.end() && itr2 != _children.end())
        std::iter_swap(itr1, itr2);
}

void Node::swapChildren(Node::Ptr child1, Node::Ptr child2) {

    auto itr1 = std::find(_children.begin(), _children.end(), child1);
    if (itr1 == _children.end())
        return;
    auto itr2 = std::find(_children.begin(), _children.end(), child2);
    if (itr2 == _children.end())
        return;
    std::iter_swap(itr1, itr2);
}

std::vector<Node*> Node::getNodesInScope() {
    mo::Mutex_t::scoped_lock lock(*_mtx);
    std::vector<Node*>       nodes;
    if (_parents.size())
        _parents[0]->getNodesInScope(nodes);
    return nodes;
}

Node* Node::getNodeInScope(const std::string& name) {
    mo::Mutex_t::scoped_lock lock(*_mtx);
    if (name == getTreeName())
        return this;
    auto current_children = _children;
    lock.unlock();
    for (auto& child : current_children) {
        auto result = child->getNodeInScope(name);
        if (result) {
            return result;
        }
    }
    return nullptr;
}

void Node::getNodesInScope(std::vector<Node*>& nodes) {
    // Perhaps not thread safe?

    // First travel to the root node
    mo::Mutex_t::scoped_lock lock(*_mtx);
    if (nodes.size() == 0) {
        Node* node = this;
        while (node->_parents.size()) {
            node = node->_parents[0].get();
        }
        nodes.push_back(node);
        node->getNodesInScope(nodes);
        return;
    }
    nodes.push_back(this);
    for (size_t i = 0; i < _children.size(); ++i) {
        if (_children[i] != nullptr)
            _children[i]->getNodesInScope(nodes);
    }
}

void Node::removeChild(Node::Ptr node) {
    mo::Mutex_t::scoped_lock lock(*_mtx);
    for (auto itr = _children.begin(); itr != _children.end(); ++itr) {
        if (*itr == node) {
            _children.erase(itr);
            return;
        }
    }
}
void Node::removeChild(int idx) {
    _children.erase(_children.begin() + idx);
}

void Node::removeChild(const std::string& name) {
    for (auto itr = _children.begin(); itr != _children.end(); ++itr) {
        if ((*itr)->getTreeName() == name) {
            _children.erase(itr);
            return;
        }
    }
}

void Node::removeChild(Node* node) {
    auto itr = std::find(_children.begin(), _children.end(), node);
    if (itr != _children.end()) {
        _children.erase(itr);
    }
}

void Node::removeChild(rcc::weak_ptr<Node> node) {
    auto itr = std::find(_children.begin(), _children.end(), node.get());
    if (itr != _children.end()) {
        _children.erase(itr);
    }
}

void Node::setDataStream(IDataStream* stream_) {
    if (stream_ == nullptr)
        return;
    mo::Mutex_t::scoped_lock lock(*_mtx);
    if (_data_stream && _data_stream != stream_) {
        _data_stream->removeNode(this);
    }
    _data_stream = stream_;
    this->setContext(stream_->getContext());
    setupSignals(_data_stream->getRelayManager());
    setupVariableManager(_data_stream->getVariableManager().get());
    _data_stream->addChildNode(this);
    for (auto& child : _children) {
        child->setDataStream(_data_stream.get());
    }
}

IDataStream* Node::getDataStream() {
    if (_parents.size() && _data_stream == nullptr) {
        MO_LOG(debug) << "Setting data stream from parent";
        setDataStream(_parents[0]->getDataStream());
    }
    if (_parents.size() == 0 && _data_stream == nullptr) {
        _data_stream = IDataStream::create();
        _data_stream->addNode(this);
    }
    return _data_stream.get();
}
std::shared_ptr<mo::IVariableManager> Node::getVariableManager() {
    return getDataStream()->getVariableManager();
}

std::string Node::getTreeName() {
    if (name.size() == 0) {
        name = std::string(GetTypeName()) + boost::lexical_cast<std::string>(_unique_id);
        //name_param.emitUpdate();
    }
    return name;
}
std::string Node::getTreeName() const {
    return name;
}

void Node::setTreeName(const std::string& name) {
    this->name = name;
    //name_param.emitUpdate();
    setParamRoot(name);
}

std::vector<rcc::weak_ptr<Node> > Node::getParents() const {
    return _parents;
}

void Node::Init(bool firstInit) {
    //ui_collector::set_node_name(getFullTreeName());
    // Node init should be called first because it is where implicit parameters should be setup
    // Then in ParmaeteredIObject, the implicit parameters will be added back to the _parameter vector
    IMetaObject::Init(firstInit);
    nodeInit(firstInit);
    if (!firstInit) {
        if (_data_stream) {
            this->setContext(_data_stream->getContext());
            setupSignals(_data_stream->getRelayManager());
            setupVariableManager(_data_stream->getVariableManager().get());
            _data_stream->addChildNode(this);
        }
    }
}

void Node::nodeInit(bool firstInit) {
}

void Node::postSerializeInit() {
    Algorithm::postSerializeInit();
    std::string name = getTreeName();
    for (auto& child : _algorithm_components) {
        auto params = child->getAllParams();

        for (auto param : params) {
            param->setTreeRoot(name);
        }
    }
}

void Node::Serialize(ISimpleSerializer* pSerializer) {
    MO_LOG(info) << "RCC serializing " << getTreeName();
    Algorithm::Serialize(pSerializer);
    SERIALIZE(_children);
    SERIALIZE(_parents);
    SERIALIZE(_pimpl_node);
    SERIALIZE(_data_stream);
    SERIALIZE(name);
    SERIALIZE(_unique_id)
}

void Node::addParent(Node* parent_) {
    mo::Mutex_t::scoped_lock lock(*_mtx);
    if (std::find(_parents.begin(), _parents.end(), parent_) != _parents.end())
        return;
    _parents.push_back(parent_);
    lock.unlock();
    parent_->addChild(this);
}

void Node::setUniqueId(int id) {
    _unique_id = id;
    setParamRoot(getTreeName());
}

mo::IParam* Node::addParameter(std::shared_ptr<mo::IParam> param) {
    auto result = mo::IMetaObject::addParam(param);
    if (result) {
        result->setTreeRoot(this->getTreeName());
    }
    return result;
}

mo::IParam* Node::addParameter(mo::IParam* param) {
    auto result = mo::IMetaObject::addParam(param);
    if (result) {
        result->setTreeRoot(this->getTreeName());
    }
    return result;
}
