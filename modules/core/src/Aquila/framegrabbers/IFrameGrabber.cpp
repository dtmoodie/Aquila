#include "Aquila/framegrabbers/IFrameGrabber.hpp"
#include "Aquila/core/IGraph.hpp"
#include "Aquila/framegrabbers//FrameGrabberInfo.hpp"
#include "Aquila/utilities/sorting.hpp"
#include "RuntimeObjectSystem/ISimpleSerializer.h"
#include <MetaObject/logging/logging.hpp>
#include <MetaObject/logging/profiling.hpp>
#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/object/detail/IMetaObjectImpl.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include <future>
using namespace aq;
using namespace aq::nodes;

int GrabberInfo::canLoad(const std::string& /*path*/) const
{
    return 0;
}

void GrabberInfo::listPaths(std::vector<std::string>& /*paths*/) const
{
    return;
}

int GrabberInfo::timeout() const
{
    return 1000;
}

bool IGrabber::processImpl()
{
    return grab();
}

int FrameGrabberInfo::loadTimeout() const
{
    return 1000;
}

int FrameGrabberInfo::canLoadPath(const std::string& path) const
{
    // Check all grabbers
    auto factory = mo::MetaObjectFactory::instance();
    auto constructors = factory->getConstructors(IGrabber::getHash());
    int max = 0;
    for (auto constructor : constructors)
    {
        const IObjectInfo* info = constructor->GetObjectInfo();
        const GrabberInfo* grabber_info = dynamic_cast<const GrabberInfo*>(info);
        if (grabber_info)
        {
            int p = grabber_info->canLoad(path);
            max = std::max(max, p);
        }
    }
    return max;
}

std::vector<std::string> FrameGrabberInfo::listLoadablePaths() const
{
    return std::vector<std::string>();
}

std::string FrameGrabberInfo::Print(IObjectInfo::Verbosity verbosity) const
{
    std::stringstream ss;
    ss << NodeInfo::Print(verbosity);
    auto docs = listLoadablePaths();
    if (docs.size())
    {
        ss << "-- Loadable paths \n";
    }
    for (const auto& doc : docs)
    {
        ss << doc << "\n";
    }
    return ss.str();
}

std::vector<std::pair<std::string, std::string>> IFrameGrabber::listAllLoadableDocuments()
{
    std::vector<std::pair<std::string, std::string>> output;
    auto factory = mo::MetaObjectFactory::instance();
    const auto hash = aq::nodes::IFrameGrabber::getHash();
    auto constructors = factory->getConstructors(hash);
    for (auto constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if (auto fg_info = dynamic_cast<const FrameGrabberInfo*>(info))
        {
            auto docs = fg_info->listLoadablePaths();
            // output.insert(output.end(), devices.begin(), devices.end());
            for (const auto& doc : docs)
            {
                output.emplace_back(doc, std::string(fg_info->GetObjectName()));
            }
        }
    }
    return output;
}

rcc::shared_ptr<IFrameGrabber> IFrameGrabber::create(const std::string& uri, const std::string& preferred_loader)
{
    auto factory = mo::MetaObjectFactory::instance();
    const auto hash = aq::nodes::IFrameGrabber::getHash();
    auto constructors = factory->getConstructors(hash);
    std::vector<IObjectConstructor*> valid_constructors;
    std::vector<int> valid_constructor_priority;
    for (auto constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if (auto fg_info = dynamic_cast<const FrameGrabberInfo*>(info))
        {
            int priority = fg_info->canLoadPath(uri);
            MO_LOG(debug, "{} priority: {}", fg_info->getDisplayName(), priority);
            if (priority != 0)
            {
                valid_constructors.push_back(constructor);
                valid_constructor_priority.push_back(priority);
            }
        }
    }
    if (valid_constructors.empty())
    {
        auto f = [&constructors]() -> std::string {
            std::stringstream ss;
            for (auto& constructor : constructors)
            {
                ss << constructor->GetName() << ", ";
            }
            return ss.str();
        };

        MO_LOG(warn, "No valid frame grabbers for {} framegrabbers: {}", uri, f());
        return rcc::shared_ptr<IFrameGrabber>();
    }

    const bool ascending = false;
    auto idx = indexSort(valid_constructor_priority, ascending);
    if (preferred_loader.size())
    {
        for (size_t i = 0; i < valid_constructors.size(); ++i)
        {
            if (preferred_loader == valid_constructors[i]->GetName())
            {
                idx.insert(idx.begin(), i);
                break;
            }
        }
    }

    for (size_t i = 0; i < idx.size(); ++i)
    {
        auto fg = rcc::shared_ptr<IFrameGrabber>(valid_constructors[idx[i]]->Construct());
        auto fg_info = dynamic_cast<const FrameGrabberInfo*>(valid_constructors[idx[i]]->GetObjectInfo());
        fg->Init(true);

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
        obj->document = uri;
        auto future = obj->promise.get_future();
        static std::vector<boost::thread*> connection_threads;
        // TODO cleanup the connection threads
        MO_LOG(info,
               "Attempting to load {} with frame_grabber: {} with priority: {}",
               uri,
               fg->GetTypeName(),
               valid_constructor_priority[idx[static_cast<size_t>(i)]]);
        mo::IAsyncStream::Ptr_t stream = mo::IAsyncStream::current();
        boost::thread* connection_thread = new boost::thread([obj, stream]() -> void {
            mo::IAsyncStream::setCurrent(stream);
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
                MO_LOG(info,
                       "Loaded {} with frame_grabber: {} with priority: {}",
                       uri,
                       fg->GetTypeName(),
                       valid_constructor_priority[idx[static_cast<size_t>(i)]]);
                delete connection_thread;
                fg->loaded_document.push_back(uri);
                return fg; // successful load
            }

            MO_LOG(warn, "Unable to load {} with {}", uri, fg_info->GetObjectName());
        }
        else // timeout
        {
            MO_LOG(warn,
                   "Timeout while loading {} with {}  after waiting {} ms",
                   uri,
                   fg_info->GetObjectName(),
                   timeout);
            connection_threads.push_back(connection_thread);
        }
    }
    return rcc::shared_ptr<IFrameGrabber>();
}

void IFrameGrabber::on_loaded_document_modified(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&)
{
    loadData(loaded_document);
}

void IFrameGrabber::restart()
{
    auto docs = loaded_document;
    Init(true);
    loadData(docs);
}

bool IFrameGrabber::loadData(std::string /*path*/)
{
    return false;
}

bool IFrameGrabber::loadData(std::vector<std::string> path)
{
    for (const auto& p : path)
    {
        loadData(p);
    }
    return false;
}
// MO_REGISTER_CLASS(IFrameGrabber)

void FrameGrabber::addComponent(const rcc::weak_ptr<IAlgorithm>& component)
{
    IFrameGrabber::addComponent(component);
    auto typed = component.lock().DynamicCast<IGrabber>();
    if (typed)
    {
        _grabbers.push_back(typed);
    }
}

bool FrameGrabber::loadData(std::string path)
{
    for (auto& grabber : _grabbers)
    {
        const GrabberInfo* ptr = dynamic_cast<const GrabberInfo*>(grabber->GetConstructor()->GetObjectInfo());
        if (ptr->canLoad(path) > 0)
        {
            return grabber->loadData(path);
        }
    }
    // find a new grabber to load this
    auto factory = mo::MetaObjectFactory::instance();
    const auto hash = IGrabber::getHash();
    auto constructors = factory->getConstructors(hash);
    std::map<int, IObjectConstructor*> priorities;
    for (auto constructor : constructors)
    {
        if (const GrabberInfo* ptr = dynamic_cast<const GrabberInfo*>(constructor->GetObjectInfo()))
        {
            int p = ptr->canLoad(path);
            if (p > 0)
            {
                priorities[p] = constructor;
            }
        }
    }
    if (priorities.size())
    {
        rcc::shared_ptr<IObject> obj = priorities.rbegin()->second->Construct();
        if (obj)
        {
            auto typed = obj.DynamicCast<IGrabber>();
            typed->Init(true);
            if (typed)
            {
                if (typed->loadData(path))
                {
                    addComponent(typed);
                    return true;
                }
            }
        }
    }
    return false;
}

bool FrameGrabber::processImpl()
{
    for (auto& grabber : _grabbers)
    {
        grabber->grab();
    }
    this->setModified();
    return true;
}

Priority_t FrameGrabber::canLoadPath(const std::string& path)
{
    auto factory = mo::MetaObjectFactory::instance();
    const auto hash = IGrabber::getHash();
    auto ctrs = factory->getConstructors(hash);
    int32_t priority = 0;
    for (auto ctr : ctrs)
    {
        auto ptr = dynamic_cast<const IGrabber::InterfaceInfo*>(ctr->GetObjectInfo());
        if (ptr)
        {
            const auto p = ptr->canLoad(path);
            priority = std::max(p, priority);
        }
    }
    return priority;
}

std::vector<std::string> FrameGrabber::listLoadablePaths()
{
    auto factory = mo::MetaObjectFactory::instance();
    const auto hash = IGrabber::getHash();
    auto ctrs = factory->getConstructors(hash);
    std::vector<std::string> output;
    for (auto ctr : ctrs)
    {
        auto ptr = dynamic_cast<const IGrabber::InterfaceInfo*>(ctr->GetObjectInfo());
        if (ptr)
        {
            ptr->listPaths(output);
        }
    }
    return output;
}

Timeout_t FrameGrabber::loadTimeout()
{
    auto factory = mo::MetaObjectFactory::instance();
    const auto hash = IGrabber::getHash();
    auto ctrs = factory->getConstructors(hash);
    Timeout_t max{0};
    for (auto ctr : ctrs)
    {
        auto ptr = dynamic_cast<const IGrabber::InterfaceInfo*>(ctr->GetObjectInfo());
        if (ptr)
        {
            const Timeout_t timeout = ptr->timeout();
            max = std::max<Timeout_t>(max, timeout);
        }
    }
    return max;
}

MO_REGISTER_CLASS(FrameGrabber)
