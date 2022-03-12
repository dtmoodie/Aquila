#include "Aquila/nodes/Node.hpp"
#include "Aquila/core/IGraph.hpp"
#include "Aquila/nodes/NodeFactory.hpp"
#include "Aquila/nodes/NodeInfo.hpp"
#include "MetaObject/params/MetaParam.hpp"
#include <Aquila/rcc/external_includes/cv_videoio.hpp>
#include <Aquila/utilities/container.hpp>

#include <MetaObject/core/SystemTable.hpp>

#include "RuntimeObjectSystem/ISimpleSerializer.h"
#include "RuntimeObjectSystem/RuntimeInclude.h"
#include "RuntimeObjectSystem/RuntimeSourceDependency.h"

#include <MetaObject/logging/callstack.hpp>
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/params/IParam.hpp>
#include <MetaObject/params/ISubscriber.hpp>
#include <MetaObject/thread/ThreadRegistry.hpp>

#include <boost/date_time.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/stacktrace/stacktrace.hpp>
#include <boost/thread.hpp>

#include <thrust/system/system_error.h>

#include <future>
#include <regex>

using namespace aq;
using namespace aq::nodes;

std::string NodeInfo::Print(IObjectInfo::Verbosity verbosity) const
{
    return mo::IMetaObjectInfo::Print(verbosity);
}

std::vector<std::string> INode::listConstructableNodes(const std::string& filter)
{
    auto constructors = mo::MetaObjectFactory::instance()->getConstructors(getHash());
    std::vector<std::string> output;
    for (IObjectConstructor* constructor : constructors)
    {
        if (filter.size())
        {
            if (std::string(constructor->GetName()).find(filter) != std::string::npos)
            {
                output.emplace_back(constructor->GetName());
            }
        }
        else
        {
            output.emplace_back(constructor->GetName());
        }
    }
    return output;
}

Node::Node() = default;

Node::~Node() = default;

bool Node::connectInput(const std::string& input_name,
                        mo::IMetaObject* output_object,
                        const std::string& output_name,
                        mo::BufferFlags type)
{
    auto input = this->getInput(input_name);
    if (!input)
    {
        return false;
    }

    mo::IPublisher* output = nullptr;
    if (output_name.empty())
    {
        output = output_object->getOutput(input_name);
        if (!output)
        {
            output = output_object->getOutput("output");
            if (!output || !input->acceptsPublisher(*output))
            {
                auto all_outputs = output_object->getOutputs();
                for (size_t i = 0; i < all_outputs.size(); ++i)
                {
                    if (input->acceptsPublisher(*all_outputs[i]))
                    {
                        if (output == nullptr)
                        {
                            output = all_outputs[i];
                        }
                        else
                        {
                            // multiple possible output variables are accepted by this input :/
                            // cannot deduce
                            LOG_ALGO(
                                warn,
                                "Could not deduce which output to connect to {} tried 'output', '{}' and any other "
                                "parameters with type {} however multiple parameters exist with this type",
                                input_name,
                                input_name,
                                input->getInputTypes());
                            return false;
                        }
                    }
                }
            }
        }
    }
    else
    {
        output = output_object->getOutput(output_name);
    }
    auto node = dynamic_cast<INode*>(output_object);

    if (output && input)
    {
        if (Algorithm::connectInput(input, output_object, output, type))
        {
            if (node)
            {
                node->addChild(*this);
            }

            return true;
        }

        return false;
    }
    if (output == nullptr)
    {
        auto outputs = output_object->getOutputs();
        auto f = [&outputs]() -> std::string {
            std::stringstream ss;
            for (auto& output : outputs)
            {
                ss << output->getName() << " ";
            }
            return ss.str();
        };
        LOG_ALGO(debug,
                 "Unable to find output with name '{}' in node '{}'.  Existing outputs: {}",
                 output_name,
                 node->getName(),
                 f());
    }
    if (input == nullptr)
    {
        auto outputs = node->getInputs();
        auto f = [&outputs]() -> std::string {
            std::stringstream ss;
            for (auto& output : outputs)
            {
                ss << output->getName() << " ";
            }
            return ss.str();
        };
        LOG_ALGO(debug,
                 "Unable to find input with name '{}' in node '{}'. Existing inputs: {}",
                 input_name,
                 this->getName(),
                 f());
    }
    return false;
}

bool Node::connectInput(mo::ISubscriber* input,
                        IMetaObject* output_object,
                        mo::IPublisher* output_param,
                        mo::BufferFlags type)
{
    if (Algorithm::connectInput(input, output_object, output_param, type))
    {
        auto node = dynamic_cast<INode*>(output_object);
        if (node)
        {
            addParent(*node);
        }
        sig_input_changed(this, input);
        return true;
    }
    else
    {
        return false;
    }
}

void Node::onParamUpdate(const mo::IParam& param, mo::Header header, mo::UpdateFlags fg, mo::IAsyncStream* stream)
{
    Algorithm::onParamUpdate(param, header, fg, stream);
    if (param.checkFlags(mo::ParamFlags::kCONTROL))
    {
        m_modified = true;
    }
    if (m_disable_due_to_errors && param.checkFlags(mo::ParamFlags::kCONTROL))
    {
        m_throw_count--;
        m_disable_due_to_errors = false;
    }
}

bool Node::process(mo::IAsyncStream& stream)
{
    LOG_ALGO(trace, "{} -- process", getName());
    ++m_iterations_since_execution;
    if (m_iterations_since_execution % 100 == 0)
    {
        const std::string reason = (m_last_execution_failure_reason ? m_last_execution_failure_reason : "");
        LOG_ALGO(
            debug, "{} has not executed in {} iterations due to {}", getName(), m_iterations_since_execution, reason);
    }
    bool can_process = true;
    if (getEnabled() && m_disable_due_to_errors == false)
    {
        mo::Lock_t lock(getMutex());
        {
            LOG_ALGO(trace, "{} checking inputs", getName());
            auto input_state = checkInputs();
            if (input_state == Algorithm::InputState::kNONE_VALID)
            {
                LOG_ALGO(trace, "{} no valid inputs", getName());
                m_last_execution_failure_reason = "No valid inputs";
                can_process = false;
            }
            if (input_state == Algorithm::InputState::kNOT_UPDATED)
            {
                LOG_ALGO(trace, "{} inputs not updated", getName());
                m_last_execution_failure_reason = "Inputs not updated and parameters not updated";
                can_process = false;
            }
        }
        m_modified = false;
        static const uint32_t exception_try_count = 10;
        try
        {
            m_last_execution_failure_reason = "Exception thrown";
            if (can_process)
            {
                LOG_ALGO(trace, "{} calling processImpl", getName());
                mo::ScopedProfile profiler(this->getName().c_str());
                stream.makeCurrent();
                if (!processImpl(stream))
                {
                    m_iterations_since_execution = 0;
                }
                else
                {
                    LOG_ALGO(trace, "{} clearing modified flag", getName());
                    clearModifiedControlParams();
                    clearModifiedInputs();
                }
            }
        }
        catch (thrust::system_error& e)
        {
            LOG_ALGO(error, "{}", e.what());
            ++m_throw_count;
            if (m_throw_count > exception_try_count)
            {
                m_disable_due_to_errors = true;
            }
        }
        catch (boost::thread_resource_error& err)
        {
            LOG_ALGO(error, "{}", err.what());
            ++m_throw_count;
            if (m_throw_count > exception_try_count)
            {
                m_disable_due_to_errors = true;
            }
        }
        catch (boost::thread_interrupted& err)
        {
            LOG_ALGO(error, "Thread interrupted");
            // Needs to pass this back up to the chain to the processing thread.
            // That way it knowns it needs to exit this thread
            throw err;
        }
        catch (boost::thread_exception& err)
        {
            LOG_ALGO(error, "{}", err.what());
            ++m_throw_count;
            if (m_throw_count > exception_try_count)
                m_disable_due_to_errors = true;
        }
        catch (cv::Exception& err)
        {
            LOG_ALGO(error, "{}", err.what());
            ++m_throw_count;
            if (m_throw_count > exception_try_count)
            {
                m_disable_due_to_errors = true;
            }
        }
        catch (const boost::exception& /*err*/)
        {
            LOG_ALGO(error, "Boost error");
            ++m_throw_count;
            if (m_throw_count > exception_try_count)
            {
                m_disable_due_to_errors = true;
            }
        }
        catch (const mo::IExceptionWithCallstack& exception)
        {
            const auto& bt = exception.getCallstack();
            LOG_ALGO(error, "{}", boost::stacktrace::detail::to_string(&bt.as_vector()[0], bt.size()));
            ++m_throw_count;
            if (m_throw_count > exception_try_count)
                m_disable_due_to_errors = true;
        }
        catch (std::exception& err)
        {
            LOG_ALGO(error, "{}", err.what());
            ++m_throw_count;
            if (m_throw_count > exception_try_count)
                m_disable_due_to_errors = true;
        }
        catch (...)
        {
            LOG_ALGO(error, "Unknown exception");
            ++m_throw_count;
            if (m_throw_count > exception_try_count)
            {
                m_disable_due_to_errors = true;
            }
        }
        if(can_process)
        {
            stream.noLongerCurrent();
        }
        m_last_execution_failure_reason = nullptr;
        m_iterations_since_execution = 0;
    }

    return processChildren(stream);
}

bool Node::processChildren(mo::IAsyncStream& stream)
{
    LOG_ALGO(trace, "{} processing children", getName());
    auto children = getChildren();
    for (auto& child : children)
    {
        if (child)
        {
            auto child_stream = child->getStream();
            if (child_stream)
            {
                // TODO new threading model
                if (child_stream->threadId() == stream.threadId())
                {
                    child->process(stream);
                }
            }
            else
            {
                child->process(stream);
            }
        }
    }
    return true;
}

void INode::reset()
{
    Init(false);
}

void Node::addChild(Node::Ptr child)
{
    auto my_stream = getStream();

    // TODO new threading
    if (my_stream && (mo::getThisThread() != my_stream->threadId()))
    {
        /*std::future<Ptr> result;
        std::promise<Ptr> promise;
        auto thread = _ctx.get()->thread_id;
        mo::ThreadSpecificQueue::push(
            std::bind([this, &promise, child]() { promise.set_value(this->addChild(child)); }), thread, this);
        result = promise.get_future();
        result.wait();
        return result.get();*/
    }
    if (child == nullptr)
    {
        return;
    }
    if (aq::contains(m_children, child))
    {
        return;
    }
    // This can happen based on a bad user config
    if (child == this)
    {
        return;
    }
    int count = 0;
    for (size_t i = 0; i < m_children.size(); ++i)
    {
        if (m_children[i] && m_children[i]->GetTypeName() == child->GetTypeName())
        {
            ++count;
        }
    }
    auto stream = getStream();
    child->setStream(stream);
    child->setGraph(getGraph());
    child->addParent(*this);
    child->setUniqueId(count);
    LOG_ALGO(trace, "[{}] Adding child {}", getName(), child->getName());
    m_children.push_back(std::move(child));
    return;
}

Node::Ptr Node::getChild(const std::string& treeName)
{
    mo::Lock_t lock(getMutex());
    for (size_t i = 0; i < m_children.size(); ++i)
    {
        if (m_children[i]->getName() == treeName)
        {
            return m_children[i];
        }
    }
    for (size_t i = 0; i < m_children.size(); ++i)
    {
        if (m_children[i]->getName() == treeName)
        {
            return m_children[i];
        }
    }
    return Node::Ptr();
}

Node::Ptr Node::getChild(const int& index)
{
    return m_children[index];
}
std::vector<Node::Ptr> Node::getChildren()
{
    mo::Lock_t lock(getMutex());
    return m_children;
}

void Node::removeChild(int idx)
{
    m_children.erase(m_children.begin() + idx);
}

void Node::removeChild(const std::string& name)
{
    for (auto itr = m_children.begin(); itr != m_children.end(); ++itr)
    {
        if ((*itr)->getName() == name)
        {
            m_children.erase(itr);
            return;
        }
    }
}

void Node::removeChild(const INode* node)
{
    std::remove(m_children.begin(), m_children.end(), node);
    for (auto& child : m_children)
    {
        child->removeChild(node);
    }
}

void Node::setGraph(rcc::weak_ptr<IGraph> graph_)
{
    auto graph = graph_.lock();
    if (graph == nullptr)
    {
        return;
    }
    mo::Lock_t lock(getMutex());
    auto _graph = m_graph.lock();
    if (_graph && (_graph != graph))
    {
        _graph->removeNode(this);
    }
    m_graph = graph;
    setStream(graph->getStream());
    setupSignals(graph->getRelayManager());
    setupParamServer(graph->getParamServer());
    graph->addChildNode(*this);
    for (auto& child : m_children)
    {
        child->setGraph(graph);
    }
}

rcc::shared_ptr<IGraph> Node::getGraph()
{
    auto graph = m_graph.lock();
    if (m_parents.size() && graph == nullptr)
    {
        rcc::shared_ptr<INode> parent = m_parents[0].lock();
        if (parent)
        {
            LOG_ALGO(debug, "Setting graph from parent");
            graph = parent->getGraph();
            setGraph(graph);
        }
    }
    if (graph == nullptr)
    {
        LOG_ALGO(warn, "Must belong to a graph");
    }

    return graph;
}

std::shared_ptr<mo::IParamServer> Node::getParamServer()
{
    auto graph = getGraph();
    if (graph)
    {
        return graph->getParamServer();
    }
    return {};
}

std::string Node::getName() const
{
    mo::Lock_t lock(getMutex());
    if (m_name.empty())
    {
        m_name = std::string(GetTypeName()) + boost::lexical_cast<std::string>(m_unique_id);
    }
    return m_name;
}

void Node::setName(const std::string& name)
{
    mo::Lock_t lock(getMutex());
    m_name = name;
    setParamRoot(name);
}

std::vector<INode::WeakPtr> Node::getParents() const
{
    return m_parents;
}

void Node::Init(bool firstInit)
{
    Algorithm::Init(firstInit);
    nodeInit(firstInit);

    if (!firstInit)
    {
        auto graph = m_graph.lock();
        if (graph)
        {
            this->setStream(graph->getStream());
            setupSignals(graph->getRelayManager());
            setupParamServer(graph->getParamServer());
            graph->addChildNode(*this);
        }
    }
}

void Node::nodeInit(bool /*firstInit*/)
{
}

void Node::postSerializeInit()
{
    Algorithm::postSerializeInit();
    std::string name = getName();
    auto components = Algorithm::getComponents();
    for (auto& child : components)
    {
        auto shared = child.lock();
        if (shared)
        {
            auto params = shared->getParams();
            for (auto param : params)
            {
                param->setTreeRoot(name);
            }
        }
    }
}

void Node::Serialize(ISimpleSerializer* pSerializer)
{
    LOG_ALGO(info, "RCC serializing {}", getName());
    Algorithm::Serialize(pSerializer);
    // TODO serialize private members
}

void Node::addParent(WeakPtr parent_)
{
    auto shared = parent_.lock();
    if (shared)
    {
        {
            mo::Lock_t lock(getMutex());
            if (aq::contains(m_parents, shared))
            {
                return;
            }
            m_parents.push_back(shared);
        }
        shared->addChild(*this);
    }
}

void Node::setUniqueId(int id)
{
    m_unique_id = id;
    setParamRoot(getName());
}

void Node::addParam(std::shared_ptr<mo::IParam> param)
{
    MO_ASSERT(param);
    Algorithm::addParam(param);
    param->setTreeRoot(this->getName());
}

void Node::addComponent(const rcc::weak_ptr<IAlgorithm>& component)
{
    Algorithm::addComponent(component);
    auto shared = component.lock();
    if (shared)
    {
        auto params = shared->getParams();
        for (auto param : params)
        {
            param->setTreeRoot(this->getName());
        }
    }
}

void Node::addParam(mo::IParam& param)
{
    Algorithm::addParam(param);
    param.setTreeRoot(this->getName());
}

bool Node::getModified() const
{
    return m_modified;
}

void Node::setModified(bool val)
{
    m_modified = val;
}
