#include <Aquila/core/Graph.hpp>
#include <Aquila/core/IGraph.hpp>

#include <Aquila/framegrabbers/IFrameGrabber.hpp>
#include <Aquila/nodes/Node.hpp>
#include <Aquila/nodes/NodeFactory.hpp>
#include <Aquila/utilities/sorting.hpp>

#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/object/RelayManager.hpp>
#include <MetaObject/params/ParamServer.hpp>
#include <MetaObject/signals/TSlot.hpp>
#include <MetaObject/thread/ThreadInfo.hpp>
#include <MetaObject/thread/ThreadPool.hpp>

#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <ce/hash.hpp>

#include <boost/chrono.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>

#include <fstream>
#include <future>

using namespace aq;
using namespace aq::nodes;
using namespace mo;

#define CATCH_MACRO                                                                                                    \
    catch (boost::thread_resource_error & err)                                                                         \
    {                                                                                                                  \
        MO_LOG(error) << err.what();                                                                                   \
    }                                                                                                                  \
    catch (boost::thread_interrupted & err)                                                                            \
    {                                                                                                                  \
        MO_LOG(error) << "Thread interrupted";                                                                         \
        /* Needs to pass this back up to the chain to the processing thread.*/                                         \
        /* That way it knowns it needs to exit this thread */                                                          \
        throw err;                                                                                                     \
    }                                                                                                                  \
    catch (boost::thread_exception & err)                                                                              \
    {                                                                                                                  \
        MO_LOG(error) << err.what();                                                                                   \
    }                                                                                                                  \
    catch (boost::exception & err)                                                                                     \
    {                                                                                                                  \
        MO_LOG(error) << "Boost error";                                                                                \
    }                                                                                                                  \
    catch (std::exception & err)                                                                                       \
    {                                                                                                                  \
        MO_LOG(error) << err.what();                                                                                   \
    }                                                                                                                  \
    catch (...)                                                                                                        \
    {                                                                                                                  \
        MO_LOG(error) << "Unknown exception";                                                                          \
    }

// **********************************************************************
//              Graph
// **********************************************************************
Graph::Graph()
{
    auto manager = getRelayManager();
    auto global_manager = mo::singleton<mo::RelayManager>();
    if (global_manager)
    {
        global_manager->connectSlots(this);
        global_manager->connectSignals(this);
    }
    manager->connectSlots(this);
    manager->connectSignals(this);
    mo::setThisThreadName("Graph");
    auto current_stream = mo::IAsyncStream::current();
    if (current_stream == nullptr)
    {
        current_stream = mo::IAsyncStream::create();
        mo::IAsyncStream::setCurrent(current_stream);
    }
    setStream(current_stream);
}

Graph::~Graph()
{
    stop();
    m_top_level_nodes.clear();
    for (auto thread : m_connection_threads)
    {
        thread->join();
        delete thread;
    }
}

void Graph::node_updated(nodes::INode*)
{
    m_dirty_flag = true;
}

void Graph::update()
{
    m_dirty_flag = true;
}

void Graph::input_changed(nodes::INode*, mo::ISubscriber*)
{
    m_dirty_flag = true;
}

void Graph::param_updated(mo::IMetaObject*, mo::IParam* param)
{
    if (param->checkFlags(mo::ParamFlags::kCONTROL) || param->checkFlags(mo::ParamFlags::kSOURCE))
    {
        m_dirty_flag = true;
    }
    if (param->checkFlags(mo::ParamFlags::kINPUT))
    {
        auto input = dynamic_cast<mo::ISubscriber*>(param);
        if (input)
        {
            mo::IParam* input_param = input->getPublisher();
            if (input_param)
            {
                if (input_param->checkFlags(mo::ParamFlags::kBUFFER))
                {
                    auto buffer_param = dynamic_cast<mo::ISubscriber*>(input_param);
                    if (buffer_param)
                    {
                        auto src = buffer_param->getPublisher();
                        if (src && src->checkFlags(mo::ParamFlags::kSOURCE))
                        {
                            m_dirty_flag = true;
                        }
                    }
                }
            }
        }
    }
}

bool Graph::getDirty() const
{
    return m_dirty_flag;
}

void Graph::param_added(mo::IMetaObject*, mo::IParam*)
{
    m_dirty_flag = true;
}

void Graph::run_continuously(bool)
{
}

void Graph::initCustom(bool firstInit)
{
    if (firstInit)
    {
        setupSignals(getRelayManager());
    }
}

mo::IAsyncStreamPtr_t Graph::getStream() const
{
    auto ctx = IGraph::getStream();
    if (!ctx)
    {
        // ctx = this->_processing_thread.getStream();
        // MO_ASSERT(ctx) << " processing thread returned a null context";
        // setStream(ctx);
    }
    return ctx;
}

std::vector<rcc::weak_ptr<aq::nodes::INode>> Graph::getTopLevelNodes()
{
    std::vector<rcc::weak_ptr<aq::nodes::INode>> output;
    for (auto& itr : m_top_level_nodes)
    {
        output.emplace_back(itr);
    }
    return output;
}

std::shared_ptr<mo::RelayManager> Graph::getRelayManager()
{
    if (m_relay_manager == nullptr)
    {
        m_relay_manager = std::make_shared<mo::RelayManager>();
    }
    return m_relay_manager;
}

rcc::shared_ptr<IUserController> Graph::getUserController()
{
    return m_window_callback_handler;
}

std::shared_ptr<mo::IParamServer> Graph::getParamServer()
{
    if (m_param_server == nullptr)
        m_param_server = std::make_shared<mo::ParamServer>();
    return m_param_server;
}

bool Graph::loadDocument(const std::string& document, const std::string& prefered_loader)
{
    std::string file_to_load = document;
    if (file_to_load.size() == 0)
        return false;
    if (file_to_load.at(0) == '\"' && file_to_load.at(file_to_load.size() - 1) == '\"')
    {
        file_to_load = file_to_load.substr(1, file_to_load.size() - 2);
    }
    std::lock_guard<std::mutex> lock(m_nodes_mtx);

    auto constructors = mo::MetaObjectFactory::instance()->getConstructors(aq::nodes::IFrameGrabber::getHash());
    std::vector<IObjectConstructor*> valid_frame_grabbers;
    std::vector<int> frame_grabber_priorities;
    if (constructors.empty())
    {
        MO_LOG(warn, "No frame grabbers found");
        return false;
    }
    for (auto& constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if (info)
        {
            auto fg_info = dynamic_cast<const nodes::FrameGrabberInfo*>(info);
            if (fg_info)
            {
                int priority = fg_info->canLoadPath(file_to_load);
                if (priority != 0)
                {
                    valid_frame_grabbers.push_back(constructor);
                    frame_grabber_priorities.push_back(priority);
                }
            }
        }
    }

    if (valid_frame_grabbers.empty())
    {
        auto f = [&constructors]() -> std::string {
            std::stringstream ss;
            for (auto& constructor : constructors)
            {
                ss << constructor->GetName() << ", ";
            }
            return ss.str();
        };
        MO_LOG(warn, "No valid frame grabbers for {} framegrabbers: {}", file_to_load, f());

        return false;
    }
    // Pick the frame grabber with highest priority

    const bool descending = true;
    auto idx = indexSort(frame_grabber_priorities, descending);
    if (prefered_loader.size())
    {
        for (size_t i = 0; i < valid_frame_grabbers.size(); ++i)
        {
            if (prefered_loader == valid_frame_grabbers[i]->GetName())
            {
                idx.insert(idx.begin(), i);
                break;
            }
        }
    }

    for (size_t i = 0; i < idx.size(); ++i)
    {
        auto fg = rcc::shared_ptr<IFrameGrabber>(valid_frame_grabbers[idx[i]]->Construct());
        auto fg_info = dynamic_cast<const FrameGrabberInfo*>(valid_frame_grabbers[idx[i]]->GetObjectInfo());
        fg->Init(true);
        fg->setGraph(*this);
        struct thread_load_object
        {
            std::promise<bool> promise;
            rcc::shared_ptr<IFrameGrabber> fg;
            std::string document;
            void load()
            {
                promise.set_value(fg->loadData(document));
            }
        };
        auto obj = new thread_load_object();
        obj->fg = fg;
        obj->document = file_to_load;
        auto future = obj->promise.get_future();
        boost::thread* connection_thread = new boost::thread([obj]() -> void {
            try
            {
                obj->load();
            }
            catch (std::exception& e)
            {
                MO_LOG(debug, e.what());
            }

            delete obj;
        });
        const auto timeout = fg_info->loadTimeout();
        if (connection_thread->timed_join(boost::posix_time::milliseconds(timeout)))
        {
            if (future.get())
            {
                m_top_level_nodes.emplace_back(fg);
                MO_LOG(info,
                       "Loading {} with frame_grabber: {} with priority: {}",
                       file_to_load,
                       fg->GetTypeName(),
                       frame_grabber_priorities[static_cast<size_t>(idx[i])]);
                delete connection_thread;
                return true; // successful load
            }

            MO_LOG(warn, "Unable to load {} with {}", file_to_load, fg_info->GetObjectName());
        }
        else // timeout
        {
            MO_LOG(warn,
                   "Timeout while loading {} with {}  after waiting {} ms",
                   file_to_load,
                   fg_info->GetObjectName(),
                   timeout);
            m_connection_threads.push_back(connection_thread);
        }
    }
    return false;
}
bool IGraph::canLoadPath(const std::string& document)
{
    std::string doc_to_load = document;
    if (doc_to_load.size() == 0)
        return false;
    if (doc_to_load.at(0) == '\"' && doc_to_load.at(doc_to_load.size() - 1) == '\"')
    {
        doc_to_load = doc_to_load.substr(1, doc_to_load.size() - 2);
    }

    auto constructors = mo::MetaObjectFactory::instance()->getConstructors(aq::nodes::IFrameGrabber::getHash());
    std::vector<IObjectConstructor*> valid_frame_grabbers;
    std::vector<int> frame_grabber_priorities;
    if (constructors.empty())
    {
        MO_LOG(warn, "No frame grabbers found");
        return false;
    }
    for (auto& constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if (info)
        {
            auto fg_info = dynamic_cast<const FrameGrabberInfo*>(info);
            if (fg_info)
            {
                int priority = fg_info->canLoadPath(doc_to_load);
                if (priority != 0)
                {
                    MO_LOG(debug, "{} can load document", fg_info->GetObjectName());
                    valid_frame_grabbers.push_back(constructor);
                    frame_grabber_priorities.push_back(priority);
                }
            }
        }
    }
    return !valid_frame_grabbers.empty();
}

std::vector<rcc::shared_ptr<nodes::INode>> Graph::getNodes() const
{
    return m_top_level_nodes;
}

std::vector<rcc::shared_ptr<nodes::INode>> Graph::getAllNodes() const
{
    std::vector<rcc::shared_ptr<nodes::INode>> output;
    for (auto& child : m_child_nodes)
    {
        auto shared = child.lock();
        if (shared)
        {
            output.emplace_back(std::move(shared));
        }
    }
    return output;
}

std::vector<rcc::shared_ptr<nodes::INode>> Graph::addNode(const std::string& nodeName)
{
    return aq::NodeFactory::Instance()->addNode(nodeName, this);
}

void Graph::addNode(const rcc::shared_ptr<nodes::INode>& node)
{
    node->setGraph(*this);
    auto name = node->getName();
    if (name.empty())
    {
        std::string node_name = node->GetTypeName();
        int count = 0;
        for (size_t i = 0; i < m_top_level_nodes.size(); ++i)
        {
            if (m_top_level_nodes[i] && m_top_level_nodes[i]->GetTypeName() == node_name)
                ++count;
        }
        node->setUniqueId(count);
    }
    m_top_level_nodes.push_back(node);
    m_dirty_flag = true;
}

void Graph::removeNode(const nodes::INode* node)
{
    std::lock_guard<std::mutex> lock(m_nodes_mtx);
    // probably need to traverse the graph and remove any other instances of this node
    std::remove(m_top_level_nodes.begin(), m_top_level_nodes.end(), node);
    std::remove(m_child_nodes.begin(), m_child_nodes.end(), node);
    for (auto& child : m_top_level_nodes)
    {
        if (child)
        {
            child->removeChild(node);
        }
    }
}

void Graph::addChildNode(rcc::shared_ptr<nodes::INode> node)
{
    std::lock_guard<std::mutex> lock(m_nodes_mtx);
    if (std::find(m_child_nodes.begin(), m_child_nodes.end(), node.get()) != m_child_nodes.end())
        return;
    int type_count = 0;
    for (auto& child : m_child_nodes)
    {
        auto shared = child.lock();
        {
            if (shared && shared != node && shared->GetTypeName() == node->GetTypeName())
            {
                ++type_count;
            }
        }
    }
    node->setUniqueId(type_count);
    m_child_nodes.emplace_back(node);
}

void Graph::addNodeNoInit(const rcc::shared_ptr<nodes::INode>& node)
{
    std::lock_guard<std::mutex> lock(m_nodes_mtx);
    m_top_level_nodes.push_back(node);
    m_dirty_flag = true;
}

void Graph::addNodes(const std::vector<rcc::shared_ptr<nodes::INode>>& nodes)
{
    std::lock_guard<std::mutex> lock(m_nodes_mtx);
    for (auto& node : nodes)
    {
        node->setGraph(*this);
    }
    /*if (!_processing_thread.isOnThread())
    {
        std::promise<void> promise;
        std::future<void> future = promise.get_future();
        _processing_thread.pushEventQueue(std::bind([&nodes, this, &promise]() {
            for (auto& node : nodes)
            {
                addNode(node);
            }
            dirty_flag = true;
            promise.set_value();
        }));
        future.wait();
    }*/
    for (auto& node : nodes)
    {
        m_top_level_nodes.push_back(node);
    }
    m_dirty_flag = true;
}

rcc::shared_ptr<nodes::INode> Graph::getNode(const std::string& node_name)
{
    std::lock_guard<std::mutex> lock(m_nodes_mtx);
    for (const auto& node : m_top_level_nodes)
    {
        // during serialization top_level_nodes is resized thus allowing
        // for nullptr nodes until they are serialized
        if (node)
        {
            if (node->getName() == node_name)
            {
                return node;
            }
        }
    }
    for (const auto& child : m_child_nodes)
    {
        auto shared = child.lock();
        if (shared)
        {
            if (shared->getName() == node_name)
            {
                return shared;
            }
        }
    }

    return {};
}

void Graph::addVariableSink(IVariableSink* sink)
{
    m_variable_sinks.push_back(sink);
}

void Graph::removeVariableSink(IVariableSink* sink)
{
    std::remove_if(m_variable_sinks.begin(), m_variable_sinks.end(), [sink](IVariableSink* other) -> bool {
        return other == sink;
    });
}

// TODO a graph should actually just push an event when one of the underlying nodes is updated
void graphLoop(rcc::shared_ptr<Graph> graph, IAsyncStreamPtr_t stream, const uint64_t event_id)
{
    graph->process();
    stream->pushEvent([graph, stream, event_id](mo::IAsyncStream*) mutable { graphLoop(graph, stream, event_id); }, event_id);
}

void Graph::start()
{
    sig_StartThreads();
    mo::IAsyncStreamPtr_t stream = this->getStream();
    MO_ASSERT(stream);

    const auto object_id = this->GetObjectId();
    const auto event_id = ce::combineHash(object_id.m_PerTypeId, object_id.m_ConstructorId);
    rcc::shared_ptr<Graph> graph_ptr(*this);
    auto event = [graph_ptr, stream, event_id](mo::IAsyncStream*) { graphLoop(graph_ptr, stream, event_id); };
    stream->pushEvent(std::move(event), event_id);
}

void Graph::stop()
{
    sig_StopThreads();

    mo::IAsyncStreamPtr_t stream = this->getStream();
    MO_ASSERT(stream);
    size_t event_id = ce::generateHash(static_cast<const void*>(this));
    auto event = [](mo::IAsyncStream*) {};
    stream->pushEvent(std::move(event), event_id);
}

int Graph::process()
{
    mo::IAsyncStream::Ptr_t stream = this->getStream();
    MO_ASSERT(stream != nullptr);
    if (m_dirty_flag /* || run_continuously == true*/)
    {
        m_dirty_flag = false;
        mo::ScopedProfile profile_nodes("Processing nodes");
        MO_LOG(trace, "-------------------------------- Processing all nodes");
        for (auto& node : m_top_level_nodes)
        {
            node->process(*stream);
        }
        if (m_dirty_flag)
        {
            return 1;
        }
    }
    else
    {
        return 10;
    }
    return 10;
}

IGraph::Ptr_t IGraph::create(const std::string& document, const std::string& preferred_frame_grabber)
{
    auto stream = Graph::create();
    if (!document.empty() || !preferred_frame_grabber.empty())
    {
        auto fg = IFrameGrabber::create(document, preferred_frame_grabber);
        if (fg)
        {
            stream->addNode(fg);
            return stream;
        }
    }
    return stream;
}

Graph::IObjectContainer* Graph::getObjectContainer(mo::TypeInfo type) const
{
    auto itr = m_objects.find(type);
    if (itr != m_objects.end())
    {
        return itr->second.get();
    }
    return nullptr;
}

void Graph::setObjectContainer(mo::TypeInfo type, IObjectContainer::Ptr_t&& container)
{
    m_objects[type] = std::move(container);
}

bool Graph::saveGraph(const std::string&)
{
    return false;
}

bool Graph::loadGraph(const std::string&)
{
    return false;
}

bool Graph::waitForSignal(const std::string& name, const boost::optional<std::chrono::milliseconds>& timeout)
{
    bool received = false;
    auto mgr = this->getRelayManager();

    if (mgr->getRelayOptional<void(void)>(name))
    {
        auto relay = mgr->getRelay<void(void)>(name);
        boost::condition_variable cv;
        boost::mutex mtx;
        mo::TSlot<void(void)> slot([&cv, &received]() {
            received = true;
            cv.notify_all();
        });
        slot.connect(relay);
        if (timeout)
        {
            boost::unique_lock<boost::mutex> lock(mtx);
            cv.wait_for(
                lock,
                boost::chrono::milliseconds(std::chrono::duration_cast<std::chrono::milliseconds>(*timeout).count()));
        }
        else
        {
            boost::unique_lock<boost::mutex> lock(mtx);
            cv.wait(lock);
        }
    }
    else
    {
        MO_LOG(info, "{} not a valid signal", name);
    }
    return received;
}

MO_REGISTER_OBJECT(Graph)
