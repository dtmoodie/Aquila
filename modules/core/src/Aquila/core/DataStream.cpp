#include "Aquila/core/DataStream.hpp"
#include "Aquila/core/IDataStream.hpp"
#include "Aquila/core/Logging.hpp"
#include "Aquila/core/ParameterBuffer.hpp"
#include "Aquila/rcc/SystemTable.hpp"

#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "Aquila/nodes/Node.hpp"
#include "Aquila/nodes/NodeFactory.hpp"
#include <Aquila/gui/UiCallbackHandlers.h>
#include <Aquila/utilities/cuda/sorting.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/object/RelayManager.hpp>
#include <MetaObject/params/VariableManager.hpp>
#include <MetaObject/serialization/memory.hpp>
#include <MetaObject/signals/TSlot.hpp>
#include <MetaObject/thread/InterThread.hpp>
#include <MetaObject/thread/ThreadPool.hpp>
#include <MetaObject/thread/boost_thread.hpp>

#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <boost/chrono.hpp>
#include <boost/filesystem.hpp>
#include <boost/thread.hpp>
#include <opencv2/core.hpp>

#include <fstream>
#include <future>

using namespace aq;
using namespace aq::nodes;
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include "MetaObject/params/traits/MemoryTraits.hpp"

INSTANTIATE_META_PARAM(rcc::shared_ptr<IDataStream>);
INSTANTIATE_META_PARAM(rcc::weak_ptr<IDataStream>);

#define CATCH_MACRO                                                            \
    catch (boost::thread_resource_error & err) {                               \
        MO_LOG(error) << err.what();                                           \
    }                                                                          \
    catch (boost::thread_interrupted & err) {                                  \
        MO_LOG(error) << "Thread interrupted";                                 \
        /* Needs to pass this back up to the chain to the processing thread.*/ \
        /* That way it knowns it needs to exit this thread */                  \
        throw err;                                                             \
    }                                                                          \
    catch (boost::thread_exception & err) {                                    \
        MO_LOG(error) << err.what();                                           \
    }                                                                          \
    catch (cv::Exception & err) {                                              \
        MO_LOG(error) << err.what();                                           \
    }                                                                          \
    catch (boost::exception & err) {                                           \
        MO_LOG(error) << "Boost error";                                        \
    }                                                                          \
    catch (std::exception & err) {                                             \
        MO_LOG(error) << err.what();                                           \
    }                                                                          \
    catch (...) {                                                              \
        MO_LOG(error) << "Unknown exception";                                  \
    }

// **********************************************************************
//              DataStream
// **********************************************************************
DataStream::DataStream() {
    _sig_manager = getRelayManager();
    auto table   = PerModuleInterface::GetInstance()->GetSystemTable();
    if (table) {
        mo::RelayManager* global_signal_manager = table->getSingleton<mo::RelayManager>();
        if (!global_signal_manager) {
            global_signal_manager = mo::RelayManager::Instance();
            table->setSingleton<mo::RelayManager>(global_signal_manager);
        }
        global_signal_manager->connectSlots(this);
        global_signal_manager->connectSignals(this);
    }
    getRelayManager()->connectSlots(this);
    getRelayManager()->connectSignals(this);
    stream_id            = 0;
    _thread_id           = 0;
    std::string old_name = mo::getThisThreadName();
    mo::setThisThreadName("DataStream");
    _processing_thread = mo::ThreadPool::instance()->requestThread();
    _processing_thread.setInnerLoop(getSlot_process<int(void)>());
    this->_ctx = this->_processing_thread.getContext();
    _processing_thread.setThreadName("DataStream");
    mo::setThisThreadName(old_name);
}

void DataStream::node_updated(nodes::Node* node) {
    (void)node;
    dirty_flag = true;
}

void DataStream::update() {
    dirty_flag = true;
}
void DataStream::input_changed(nodes::Node* node, mo::InputParam* param) {
    (void)node;
    (void)param;
    dirty_flag = true;
}
void DataStream::param_updated(mo::IMetaObject* obj, mo::IParam* param) {
    (void)obj;
    if (param->checkFlags(mo::Control_e) || param->checkFlags(mo::Source_e))
        dirty_flag = true;
}

void DataStream::param_added(mo::IMetaObject* obj, mo::IParam* param) {
    (void)obj;
    (void)param;
    dirty_flag = true;
}

void DataStream::run_continuously(bool value) {
    (void)value;
}

void DataStream::initCustom(bool firstInit) {
    if (firstInit) {
        this->setupSignals(getRelayManager());
    }
}

DataStream::~DataStream() {
    stopThread();
    top_level_nodes.clear();
    relay_manager.reset();
    _sig_manager = nullptr;
    for (auto thread : connection_threads) {
        thread->join();
        delete thread;
    }
}

mo::ContextPtr_t DataStream::getContext() {
    auto ctx = IDataStream::getContext();
    if (!ctx) {
        ctx = this->_processing_thread.getContext();
        MO_ASSERT(ctx) << " processing thread returned a null context";
        setContext(ctx);
    }
    return ctx;
}

std::vector<rcc::weak_ptr<aq::nodes::Node> > DataStream::getTopLevelNodes() {
    std::vector<rcc::weak_ptr<aq::nodes::Node> > output;
    for (auto& itr : top_level_nodes) {
        output.emplace_back(itr);
    }
    return output;
}

mo::RelayManager* DataStream::getRelayManager() {
    if (relay_manager == nullptr)
        relay_manager.reset(new mo::RelayManager());
    return relay_manager.get();
}

IParameterBuffer* DataStream::getParameterBuffer() {
    if (_parameter_buffer == nullptr)
        _parameter_buffer.reset(new ParameterBuffer(10));
    return _parameter_buffer.get();
}
rcc::weak_ptr<WindowCallbackHandler> DataStream::getWindowCallbackManager() {
    if (!_window_callback_handler) {
        _window_callback_handler = WindowCallbackHandler::create();
        _window_callback_handler->setupSignals(this->getRelayManager());
    }
    return _window_callback_handler;
}

std::shared_ptr<mo::IVariableManager> DataStream::getVariableManager() {
    if (variable_manager == nullptr)
        variable_manager.reset(new mo::VariableManager());
    return variable_manager;
}

bool DataStream::loadDocument(const std::string& document, const std::string& prefered_loader) {
    std::string file_to_load = document;
    if (file_to_load.size() == 0)
        return false;
    if (file_to_load.at(0) == '\"' && file_to_load.at(file_to_load.size() - 1) == '\"') {
        file_to_load = file_to_load.substr(1, file_to_load.size() - 2);
    }
    std::lock_guard<std::mutex> lock(nodes_mtx);

    auto                             constructors = mo::MetaObjectFactory::instance()->getConstructors(aq::nodes::IFrameGrabber::s_interfaceID);
    std::vector<IObjectConstructor*> valid_frame_grabbers;
    std::vector<int>                 frame_grabber_priorities;
    if (constructors.empty()) {
        MO_LOG(warning) << "No frame grabbers found";
        return false;
    }
    for (auto& constructor : constructors) {
        auto info = constructor->GetObjectInfo();
        if (info) {
            auto fg_info = dynamic_cast<nodes::FrameGrabberInfo*>(info);
            if (fg_info) {
                int priority = fg_info->canLoadPath(file_to_load);
                if (priority != 0) {
                    valid_frame_grabbers.push_back(constructor);
                    frame_grabber_priorities.push_back(priority);
                }
            }
        }
    }

    if (valid_frame_grabbers.empty()) {
        auto f = [&constructors]() -> std::string {
            std::stringstream ss;
            for (auto& constructor : constructors) {
                ss << constructor->GetName() << ", ";
            }
            return ss.str();
        };
        MO_LOG(warning) << "No valid frame grabbers for " << file_to_load
                        << " framegrabbers: " << f();

        return false;
    }
    // Pick the frame grabber with highest priority

    auto idx = sort_index_descending(frame_grabber_priorities);
    if (prefered_loader.size()) {
        for (size_t i = 0; i < valid_frame_grabbers.size(); ++i) {
            if (prefered_loader == valid_frame_grabbers[i]->GetName()) {
                idx.insert(idx.begin(), i);
                break;
            }
        }
    }

    for (size_t i = 0; i < idx.size(); ++i) {
        auto fg      = rcc::shared_ptr<IFrameGrabber>(valid_frame_grabbers[idx[i]]->Construct());
        auto fg_info = dynamic_cast<FrameGrabberInfo*>(valid_frame_grabbers[idx[i]]->GetObjectInfo());
        fg->Init(true);
        fg->setDataStream(this);
        struct thread_load_object {
            std::promise<bool>             promise;
            rcc::shared_ptr<IFrameGrabber> fg;
            std::string                    document;
            void                           load() {
                promise.set_value(fg->loadData(document));
            }
        };
        auto obj                         = new thread_load_object();
        obj->fg                          = fg;
        obj->document                    = file_to_load;
        auto           future            = obj->promise.get_future();
        boost::thread* connection_thread = new boost::thread([obj]() -> void {
            try {
                obj->load();
            } catch (cv::Exception& e) {
                MO_LOG(debug) << e.what();
            }

            delete obj;
        });
        if (connection_thread->timed_join(boost::posix_time::milliseconds(fg_info->loadTimeout()))) {
            if (future.get()) {
                top_level_nodes.emplace_back(fg);
                MO_LOG(info) << "Loading " << file_to_load << " with frame_grabber: " << fg->GetTypeName() << " with priority: " << frame_grabber_priorities[static_cast<size_t>(idx[i])];
                delete connection_thread;
                return true; // successful load
            } else // unsuccessful load
            {
                MO_LOG(warning) << "Unable to load " << file_to_load << " with " << fg_info->GetObjectName();
            }
        } else // timeout
        {
            MO_LOG(warning) << "Timeout while loading " << file_to_load << " with " << fg_info->GetObjectName() << " after waiting " << fg_info->loadTimeout() << " ms";
            connection_threads.push_back(connection_thread);
        }
    }
    return false;
}
bool IDataStream::canLoadPath(const std::string& document) {
    std::string doc_to_load = document;
    if (doc_to_load.size() == 0)
        return false;
    if (doc_to_load.at(0) == '\"' && doc_to_load.at(doc_to_load.size() - 1) == '\"') {
        doc_to_load = doc_to_load.substr(1, doc_to_load.size() - 2);
    }

    auto                             constructors = mo::MetaObjectFactory::instance()->getConstructors(aq::nodes::IFrameGrabber::s_interfaceID);
    std::vector<IObjectConstructor*> valid_frame_grabbers;
    std::vector<int>                 frame_grabber_priorities;
    if (constructors.empty()) {
        MO_LOG(warning) << "No frame grabbers found";
        return false;
    }
    for (auto& constructor : constructors) {
        auto info = constructor->GetObjectInfo();
        if (info) {
            auto fg_info = dynamic_cast<FrameGrabberInfo*>(info);
            if (fg_info) {
                int priority = fg_info->canLoadPath(doc_to_load);
                if (priority != 0) {
                    MO_LOG(debug) << fg_info->GetObjectName() << " can load document";
                    valid_frame_grabbers.push_back(constructor);
                    frame_grabber_priorities.push_back(priority);
                }
            }
        }
    }
    return !valid_frame_grabbers.empty();
}

std::vector<rcc::shared_ptr<nodes::Node> > DataStream::getNodes() const {
    return top_level_nodes;
}
std::vector<rcc::shared_ptr<nodes::Node> > DataStream::getAllNodes() const {
    std::vector<rcc::shared_ptr<nodes::Node> > output;
    for (auto& child : child_nodes) {
        output.emplace_back(child);
    }
    return output;
}
std::vector<rcc::shared_ptr<nodes::Node> > DataStream::addNode(const std::string& nodeName) {
    return aq::NodeFactory::Instance()->addNode(nodeName, this);
}
void DataStream::addNode(rcc::shared_ptr<nodes::Node> node) {
    node->setDataStream(this);
    if (!_processing_thread.isOnThread() && _processing_thread.getIsRunning()) {
        std::promise<void> promise;
        std::future<void>  future = promise.get_future();

        _processing_thread.pushEventQueue(std::bind([&promise, node, this]() {
            rcc::shared_ptr<Node> node_ = node;
            if (std::find(top_level_nodes.begin(), top_level_nodes.end(), node) != top_level_nodes.end()) {
                promise.set_value();
                return;
            }

            if (node->name.size() == 0) {
                std::string node_name = node->GetTypeName();
                int         count     = 0;
                for (size_t i = 0; i < top_level_nodes.size(); ++i) {
                    if (top_level_nodes[i] && top_level_nodes[i]->GetTypeName() == node_name)
                        ++count;
                }
                node_->setUniqueId(count);
            }
            node_->setParamRoot(node_->getTreeName());
            top_level_nodes.push_back(node);
            dirty_flag = true;
            promise.set_value();
        }));
        future.wait();
        return;
    }
    if (std::find(top_level_nodes.begin(), top_level_nodes.end(), node) != top_level_nodes.end())
        return;
    if (node->name.size() == 0) {
        std::string node_name = node->GetTypeName();
        int         count     = 0;
        for (size_t i = 0; i < top_level_nodes.size(); ++i) {
            if (top_level_nodes[i] && top_level_nodes[i]->GetTypeName() == node_name)
                ++count;
        }
        node->setUniqueId(count);
    }
    node->setParamRoot(node->getTreeName());
    top_level_nodes.push_back(node);
    dirty_flag = true;
}
void DataStream::addChildNode(rcc::shared_ptr<nodes::Node> node) {
    std::lock_guard<std::mutex> lock(nodes_mtx);
    if (std::find(child_nodes.begin(), child_nodes.end(), node.get()) != child_nodes.end())
        return;
    int type_count = 0;
    for (auto& child : child_nodes) {
        if (child && child != node && child->GetTypeName() == node->GetTypeName())
            ++type_count;
    }
    node->setUniqueId(type_count);
    child_nodes.emplace_back(node);
}
void DataStream::removeChildNode(rcc::shared_ptr<nodes::Node> node) {
    std::lock_guard<std::mutex> lock(nodes_mtx);
    std::remove(child_nodes.begin(), child_nodes.end(), node);
}
void DataStream::addNodeNoInit(rcc::shared_ptr<nodes::Node> node) {
    std::lock_guard<std::mutex> lock(nodes_mtx);
    top_level_nodes.push_back(node);
    dirty_flag = true;
}
void DataStream::addNodes(std::vector<rcc::shared_ptr<nodes::Node> > nodes) {
    std::lock_guard<std::mutex> lock(nodes_mtx);
    for (auto& node : nodes) {
        node->setDataStream(this);
    }
    if (!_processing_thread.isOnThread()) {
        std::promise<void> promise;
        std::future<void>  future = promise.get_future();
        _processing_thread.pushEventQueue(std::bind([&nodes, this, &promise]() {
            for (auto& node : nodes) {
                addNode(node);
            }
            dirty_flag = true;
            promise.set_value();
        }));
        future.wait();
    }
    for (auto& node : nodes) {
        top_level_nodes.push_back(node);
    }
    dirty_flag = true;
}

void DataStream::removeNode(nodes::Node* node) {
    {
        std::lock_guard<std::mutex> lock(nodes_mtx);
        std::remove(top_level_nodes.begin(), top_level_nodes.end(), node);
    }

    removeChildNode(node);
}

void DataStream::removeNode(rcc::shared_ptr<nodes::Node> node) {
    {
        std::lock_guard<std::mutex> lock(nodes_mtx);
        std::remove(top_level_nodes.begin(), top_level_nodes.end(), node);
    }
    removeChildNode(node);
}

nodes::Node* DataStream::getNode(const std::string& nodeName) {
    std::lock_guard<std::mutex> lock(nodes_mtx);
    for (auto& node : top_level_nodes) {
        if (node) // during serialization top_level_nodes is resized thus allowing for nullptr nodes until they are serialized
        {
            auto found_node = node->getNodeInScope(nodeName);
            if (found_node) {
                return found_node;
            }
        }
    }

    return nullptr;
}

void DataStream::addVariableSink(IVariableSink* sink) {
    variable_sinks.push_back(sink);
}

void DataStream::removeVariableSink(IVariableSink* sink) {
    std::remove_if(variable_sinks.begin(), variable_sinks.end(), [sink](IVariableSink* other) -> bool { return other == sink; });
}
void DataStream::startThread() {
    sig_StartThreads();
    _processing_thread.start();
}

void DataStream::stopThread() {
    sig_StopThreads();
    _processing_thread.stop();
}

void DataStream::pauseThread() {
    sig_StopThreads();
    _processing_thread.stop();
}

void DataStream::resumeThread() {
    _processing_thread.start();
    sig_StartThreads();
}

int DataStream::process() {
    if (dirty_flag /* || run_continuously == true*/) {
        dirty_flag = false;
        mo::scoped_profile profile_nodes("Processing nodes", nullptr, nullptr, getContext()->getCudaStream());
        for (auto& node : top_level_nodes) {
            node->process();
        }
        if (dirty_flag) {
            return 1;
        }
    } else {
        return 10;
    }
    return 10;
}

IDataStream::Ptr IDataStream::create(const std::string& document, const std::string& preferred_frame_grabber) {
    auto stream = DataStream::create();
    if (document.size() || preferred_frame_grabber.size()) {
        auto fg = IFrameGrabber::create(document, preferred_frame_grabber);
        if (fg) {
            stream->addNode(fg);
            return stream;
        }
    }
    return stream;
}
std::unique_ptr<ISingleton>& DataStream::getSingleton(mo::TypeInfo type) {
    return _singletons[type];
}
std::unique_ptr<ISingleton>& DataStream::getIObjectSingleton(mo::TypeInfo type) {
    return _iobject_singletons[type];
}

MO_REGISTER_OBJECT(DataStream)
