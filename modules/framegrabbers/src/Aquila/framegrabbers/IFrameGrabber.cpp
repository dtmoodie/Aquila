#include "Aquila/Nodes/IFrameGrabber.hpp"
#include "Aquila/IDataStream.hpp"
#include "Aquila/utilities/sorting.hpp"
#include "Aquila/Nodes/FrameGrabberInfo.hpp"
#include <MetaObject/Logging/Log.hpp>
#include <MetaObject/Logging/Profiling.hpp>
#include <MetaObject/Detail/IMetaObjectImpl.hpp>
#include <MetaObject/MetaObjectFactory.hpp>
#include <MetaObject/Logging/Profiling.hpp>
#include "RuntimeObjectSystem/ISimpleSerializer.h"

using namespace aq;
using namespace aq::Nodes;

int GrabberInfo::CanLoad(const std::string& path) const
{
    return 0;
}
void GrabberInfo::ListPaths(std::vector<std::string>& paths) const
{
    return;
}
int GrabberInfo::Timeout() const
{
    return 1000;
}
bool IGrabber::ProcessImpl()
{
    return Grab();
}
int FrameGrabberInfo::LoadTimeout() const
{
    return 1000;
}

int FrameGrabberInfo::CanLoadPath(const std::string& path) const
{
    // Check all grabbers
    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors(IGrabber::s_interfaceID);
    int max = 0;
    for( auto constructor : constructors)
    {
        max = std::max(max, dynamic_cast<GrabberInfo*>(constructor->GetObjectInfo())->CanLoad(path));
    }
    return max;
}

std::vector<std::string> FrameGrabberInfo::ListLoadablePaths() const
{
    return std::vector<std::string>();
}
std::string FrameGrabberInfo::Print() const
{
    std::stringstream ss;
    ss << NodeInfo::Print();
    auto docs = ListLoadablePaths();
    if(docs.size())
    {
        ss << "-- Loadable paths \n";
    }
    for(const auto& doc : docs)
    {
        ss << doc << "\n";
    }
    return ss.str();
}
std::vector<std::pair<std::string, std::string>> IFrameGrabber::ListAllLoadableDocuments()
{
    std::vector<std::pair<std::string, std::string>> output;
    auto constructors = mo::MetaObjectFactory::Instance()->
            GetConstructors(aq::Nodes::IFrameGrabber::s_interfaceID);
    for(auto constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if(auto fg_info = dynamic_cast<FrameGrabberInfo*>(info))
        {
            auto docs = fg_info->ListLoadablePaths();
            //output.insert(output.end(), devices.begin(), devices.end());
            for(const auto& doc : docs)
            {
                output.emplace_back(doc, std::string(fg_info->GetObjectName()));
            }
        }
    }
    return output;
}

rcc::shared_ptr<IFrameGrabber> IFrameGrabber::Create(const std::string& uri,
                                                     const std::string& preferred_loader)
{
    auto constructors = mo::MetaObjectFactory::Instance()->
            GetConstructors(aq::Nodes::IFrameGrabber::s_interfaceID);
    std::vector<IObjectConstructor*> valid_constructors;
    std::vector<int> valid_constructor_priority;
    for(auto constructor : constructors)
    {
        auto info = constructor->GetObjectInfo();
        if(auto fg_info = dynamic_cast<FrameGrabberInfo*>(info))
        {
            int priority = fg_info->CanLoadPath(uri);
            LOG(debug) << fg_info->GetDisplayName() << " priority: " << priority;
            if(priority != 0)
            {
                valid_constructors.push_back(constructor);
                valid_constructor_priority.push_back(priority);
            }
        }
    }
    if (valid_constructors.empty())
    {
        auto f = [&constructors]()->std::string
        {
            std::stringstream ss;
            for(auto& constructor : constructors)
            {
                ss << constructor->GetName() << ", ";
            }
            return ss.str();
        };

        LOG(warning) << "No valid frame grabbers for " << uri
                     << " framegrabbers: " << f();
        return rcc::shared_ptr<IFrameGrabber>();
    }

    auto idx = sort_index_descending(valid_constructor_priority);
    if (preferred_loader.size())
    {
        for (int i = 0; i < valid_constructors.size(); ++i)
        {
            if (preferred_loader == valid_constructors[i]->GetName())
            {
                idx.insert(idx.begin(), i);
                break;
            }
        }
    }

    for (int i = 0; i < idx.size(); ++i)
    {
        auto fg = rcc::shared_ptr<IFrameGrabber>(valid_constructors[idx[i]]->Construct());
        auto fg_info = dynamic_cast<FrameGrabberInfo*>(valid_constructors[idx[i]]->GetObjectInfo());
        fg->Init(true);
        
        struct thread_load_object
        {
            std::promise<bool> promise;
            rcc::shared_ptr<IFrameGrabber> fg;
            std::string document;
            void load()
            {
                promise.set_value(fg->Load(document));
            }
        };
        
        auto obj = new thread_load_object();
        obj->fg = fg;
        obj->document = uri;
        auto future = obj->promise.get_future();
        static std::vector<boost::thread*> connection_threads;
        // TODO cleanup the connection threads

        boost::thread* connection_thread = new boost::thread([obj]()->void {
            try
            {
                obj->load();
            }
            catch (cv::Exception&e)
            {
                LOG(debug) << e.what();
            }

            delete obj;
        });
        
        if (connection_thread->timed_join(boost::posix_time::milliseconds(fg_info->LoadTimeout())))
        {
            if (future.get())
            {
                LOG(info) << "Loading " << uri << " with frame_grabber: " << fg->GetTypeName() << " with priority: " << valid_constructor_priority[idx[i]];
                delete connection_thread;
                fg->loaded_document.push_back(uri);
                return fg; // successful load
            }
            else // unsuccessful load
            {
                LOG(warning) << "Unable to load " << uri << " with " << fg_info->GetObjectName();
            }
        }
        else // timeout        
        {
            LOG(warning) << "Timeout while loading " << uri << " with " << fg_info->GetObjectName() << " after waiting " << fg_info->LoadTimeout() << " ms";
            connection_threads.push_back(connection_thread);
        }
    }
    return rcc::shared_ptr<IFrameGrabber>();
}

void IFrameGrabber::on_loaded_document_modified(mo::Context*, mo::IParameter*)
{
    Load(loaded_document);
}

void IFrameGrabber::Restart()
{
    auto docs = loaded_document;
    Init(true);
    Load(docs);
}

bool IFrameGrabber::Load(std::string path)
{
    return false;
}

bool IFrameGrabber::Load(std::vector<std::string> path)
{
    for(const auto& p : path)
    {
        Load(p);
    }
    return false;
}
//MO_REGISTER_CLASS(IFrameGrabber)

class FrameGrabber: public IFrameGrabber
{
public:
    static std::vector<std::string> ListLoadablePaths();
    MO_DERIVE(FrameGrabber, IFrameGrabber)
        MO_SLOT(bool, Load, std::string)
    MO_END;
    bool ProcessImpl();
    void AddComponent(rcc::weak_ptr<Algorithm> component);
protected:
    std::vector<IGrabber::Ptr> _grabbers;
};

void FrameGrabber::AddComponent(rcc::weak_ptr<Algorithm> component)
{
    Node::AddComponent(component);
    auto typed = component.DynamicCast<IGrabber>();
    if (typed)
    {
        _grabbers.push_back(typed);
    }
}

bool FrameGrabber::Load(std::string path)
{
    for (auto& grabber : _grabbers)
    {
        GrabberInfo* ptr = dynamic_cast<GrabberInfo*>(grabber->GetConstructor()->GetObjectInfo());
        if (ptr->CanLoad(path) > 0)
        {
            grabber->Load(path);
            return true;
        }
    }
    // find a new grabber to load this
    auto constructors = mo::MetaObjectFactory::Instance()->GetConstructors(IGrabber::s_interfaceID);
    std::map<int, IObjectConstructor*> priorities;
    for (auto constructor : constructors)
    {
        if (GrabberInfo* ptr = dynamic_cast<GrabberInfo*>(constructor->GetObjectInfo()))
        {
            int p = ptr->CanLoad(path);
            if (p > 0)
            {
                priorities[p] = constructor;
            }
        }
    }
    if (priorities.size())
    {
        IObject* obj = priorities.rbegin()->second->Construct();
        if (obj)
        {
            auto typed = dynamic_cast<IGrabber*>(obj);
            typed->Init(true);
            if (typed)
            {
                if (typed->Load(path))
                {
                    AddComponent(typed);
                    return true;
                }
            }
            else
            {
                delete typed;
            }
        }
    }
    return false;
}
bool FrameGrabber::ProcessImpl()
{
    for (auto& grabber : _grabbers)
    {
        grabber->Grab();
    }
    return true;
}

std::vector<std::string> FrameGrabber::ListLoadablePaths()
{
    auto ctrs = mo::MetaObjectFactory::Instance()->GetConstructors(IGrabber::s_interfaceID);
    std::vector<std::string> output;
    for (auto ctr : ctrs)
    {
        auto ptr = dynamic_cast<IGrabber::InterfaceInfo*>(ctr->GetObjectInfo());
        if (ptr)
        {
            ptr->ListPaths(output);
        }
    }
    return output;
}

MO_REGISTER_CLASS(FrameGrabber)
