#include "MetaObject/MetaObjectFactory.hpp"
#include "RuntimeObjectSystem.h"
#include "IObjectFactorySystem.h"
#include "MetaObject/Logging/CompileLogger.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Logging/Log.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Signals/TypedSlot.hpp"

#include <boost/filesystem.hpp>
#include <boost/date_time.hpp>
using namespace mo;

struct MetaObjectFactory::impl
{
    impl(SystemTable* table)
    {
        obj_system.Initialise(&logger, table);
    }
    RuntimeObjectSystem obj_system;
    CompileLogger logger;
    std::vector<std::string> plugins;
    TypedSignal<void(void)> on_constructor_added;
};

MetaObjectFactory::MetaObjectFactory(SystemTable* table)
{
    _pimpl = new impl(table);
}

MetaObjectFactory::~MetaObjectFactory()
{
    delete _pimpl;
}

IRuntimeObjectSystem* MetaObjectFactory::GetObjectSystem()
{
    return &_pimpl->obj_system;
}

MetaObjectFactory* MetaObjectFactory::Instance(SystemTable* table)
{
    static MetaObjectFactory g_inst(table);
    return &g_inst;
}

IMetaObject* MetaObjectFactory::Create(const char* type_name, int interface_id)
{
    auto constructor = _pimpl->obj_system.GetObjectFactorySystem()->GetConstructor(type_name);
    if(constructor)
    {
        if(interface_id != -1)
        {
            if(constructor->GetInterfaceId() != interface_id)
                return nullptr;
        }
        IObject* obj = constructor->Construct();
        if(IMetaObject* mobj = dynamic_cast<IMetaObject*>(obj))
        {
            mobj->Init(true);
            return mobj;
        }else
        {
            delete obj;
        }
    }
    return nullptr;
}
IMetaObject* MetaObjectFactory::Get(ObjectId id, const char* type_name)
{
    AUDynArray<IObjectConstructor*> constructors;
    _pimpl->obj_system.GetObjectFactorySystem()->GetAll(constructors);
    if(id.m_ConstructorId < constructors.Size() && id.m_ConstructorId >= 0)
    {
        if(strcmp(constructors[id.m_ConstructorId]->GetName(), type_name) == 0)
        {
            IObject* obj = constructors[id.m_ConstructorId]->GetConstructedObject(id.m_PerTypeId);
            if (obj)
            {
                return dynamic_cast<IMetaObject*>(obj);
            }
            else
            {
                LOG(debug) << "No object exits for " << type_name << " with instance id of " << id.m_PerTypeId;
            }
        }else
        {
            LOG(debug) << "Requested type \"" << type_name << "\" does not match constructor type \"" << constructors[id.m_ConstructorId]->GetName() << "\" for given ID";
            std::string str_type_name(type_name);
            for(int i = 0; i < constructors.Size(); ++i)
            {
                if(std::string(constructors[i]->GetName()) == str_type_name)
                {
                    IObject* obj = constructors[i]->GetConstructedObject(id.m_PerTypeId);
                    if(obj)
                    {
                        return dynamic_cast<IMetaObject*>(obj);
                    }else 
                    {
                        return nullptr; // Object just doesn't exist yet.
                    }
                }
            }
            LOG(warning) << "Requested type \"" << type_name << "\" not found";
        }
    }
    return nullptr;
}

std::vector<std::string> MetaObjectFactory::ListConstructableObjects(int interface_id) const
{
    std::vector<std::string> output;
    AUDynArray<IObjectConstructor*> constructors;
    _pimpl->obj_system.GetObjectFactorySystem()->GetAll(constructors);
    for(int i = 0; i < constructors.Size(); ++i)
    {
        if(interface_id == -1)
            output.emplace_back(constructors[i]->GetName());
        else
            if(constructors[i]->GetInterfaceId() == interface_id)
                output.emplace_back(constructors[i]->GetName());
    }
    return output;
}

std::string MetaObjectFactory::PrintAllObjectInfo(int interface_id) const
{
    std::stringstream ss;
    AUDynArray<IObjectConstructor*> constructors;
    _pimpl->obj_system.GetObjectFactorySystem()->GetAll(constructors);
    for(int i = 0; i < constructors.Size(); ++i)
    {
        if(auto info = constructors[i]->GetObjectInfo())
        {
            if(interface_id == -1)
            {
                ss << info->Print();
            }
            else
            {
                if(constructors[i]->GetInterfaceId() == interface_id)
                {
                    ss << info->Print();
                }
            }
        }
    }
    return ss.str();
}

IObjectConstructor* MetaObjectFactory::GetConstructor(const char* type_name) const
{
    return _pimpl->obj_system.GetObjectFactorySystem()->GetConstructor(type_name);
}


std::vector<IObjectConstructor*> MetaObjectFactory::GetConstructors(int interface_id) const
{
    std::vector<IObjectConstructor*> output;
    AUDynArray<IObjectConstructor*> constructors;
    _pimpl->obj_system.GetObjectFactorySystem()->GetAll(constructors);
    for(int i = 0; i < constructors.Size(); ++i)
    {
        if(interface_id == -1)
            output.emplace_back(constructors[i]);
        else
            if(constructors[i]->GetInterfaceId() == interface_id)
                output.emplace_back(constructors[i]);
    }
    return output;
}

IObjectInfo* MetaObjectFactory::GetObjectInfo(const char* type_name) const
{
    auto constructor = _pimpl->obj_system.GetObjectFactorySystem()->GetConstructor(type_name);
    if(constructor)
    {
        return constructor->GetObjectInfo();
    }
    return nullptr;
}
std::vector<IObjectInfo*> MetaObjectFactory::GetAllObjectInfo() const
{
    std::vector<IObjectInfo*> output;
    AUDynArray<IObjectConstructor*> constructors;
    _pimpl->obj_system.GetObjectFactorySystem()->GetAll(constructors);
    for(int i = 0; i < constructors.Size(); ++i)
    {
        output.push_back(constructors[i]->GetObjectInfo());
    }
    return output;
}
void MetaObjectFactory::SetupObjectConstructors(IPerModuleInterface* pPerModuleInterface)
{
    GetObjectSystem()->SetupObjectConstructors(pPerModuleInterface);
}


std::vector<std::string> MetaObjectFactory::ListLoadedPlugins() const
{
    return _pimpl->plugins;
}
int MetaObjectFactory::LoadPlugins(const std::string& path_)
{
    boost::filesystem::path path(boost::filesystem::current_path().string() + "/" + path_);
    int count = 0;
    for(boost::filesystem::directory_iterator itr(path); itr != boost::filesystem::directory_iterator(); ++itr)
    {
#ifdef _MSC_VER
        if(itr->path().extension().string() == ".dll")
#else
        if(itr->path().extension().string() == ".so")
#endif
        {
            if(LoadPlugin(itr->path().string()))
                ++count;
        }
    }
    return count;
}

#ifdef _MSC_VER
#include "Windows.h"
std::string GetLastErrorAsString()
{
    //Get the error message, if any.
    DWORD errorMessageID = ::GetLastError();
    if (errorMessageID == 0)
        return std::string(); //No error message has been recorded

    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

    std::string message(messageBuffer, size);

    //Free the buffer.
    LocalFree(messageBuffer);

    return message;
}

bool MetaObjectFactory::LoadPlugin(const std::string& fullPluginPath)
{
    static int projectCount = 0;
    LOG(info) << "Loading plugin " << fullPluginPath;
    if (!boost::filesystem::is_regular_file(fullPluginPath))
    {
        return false;
    }
    std::string plugin_name = boost::filesystem::path(fullPluginPath).stem().string();
    HMODULE handle = LoadLibrary(fullPluginPath.c_str());
    if (handle == nullptr)
    {
        auto err = GetLastError();
        LOG(debug) << "Failed to load " << plugin_name << " due to: [" << err << "] " << GetLastErrorAsString();
        _pimpl->plugins.push_back(fullPluginPath + " - failed");
        
        return false;
    }
    
    typedef IPerModuleInterface* (*moduleFunctor)();
    moduleFunctor module = (moduleFunctor)GetProcAddress(handle, "GetPerModuleInterface");
    if (module)
    {
        auto moduleInterface = module();
        moduleInterface->SetModuleFileName(fullPluginPath.c_str());
        boost::filesystem::path path(fullPluginPath);
        std::string base = path.stem().replace_extension("").string();
#ifdef _DEBUG
        base = base.substr(0, base.size() - 1);
#endif
        boost::filesystem::path config_path = path.parent_path();
        config_path += "/" + base + "_config.txt";
        int id = _pimpl->obj_system.ParseConfigFile(config_path.string().c_str());
        if(id >= 0)
        {
            moduleInterface->SetProjectIdForAllConstructors(id);
        }
        SetupObjectConstructors(moduleInterface);
    }
    _pimpl->plugins.push_back(plugin_name + " - success");
    return true;
}
#else
#include "dlfcn.h"

bool MetaObjectFactory::LoadPlugin(const std::string& fullPluginPath)
{
    LOG(info) << "Loading " << fullPluginPath;
    void* handle = dlopen(fullPluginPath.c_str(), RTLD_NOW);
    // Fallback on old module
    if(handle == nullptr)
    {
        const char *dlsym_error = dlerror();
        if (dlsym_error) {
            LOG(warning)  << dlsym_error << '\n';
            return false;
        }
    }
    typedef IPerModuleInterface* (*moduleFunctor)();

    moduleFunctor module = (moduleFunctor)dlsym(handle, "GetPerModuleInterface");
    const char *dlsym_error = dlerror();
    if (dlsym_error) {
        LOG(warning)  << dlsym_error << '\n';
        return false;
    }
    if (module == nullptr)
    {
        LOG(warning)  << "module == nullptr" << std::endl;
        return false;
    }
    IPerModuleInterface* interface = module();
    interface->SetModuleFileName(fullPluginPath.c_str());
    boost::filesystem::path path(fullPluginPath);
    std::string base = path.stem().replace_extension("").string();
#ifdef NDEBUG
    base = base.substr(3, base.size() - 3);
#else
    base = base.substr(3, base.size() - 4); // strip off the d tag on the library file
#endif
    boost::filesystem::path config_path = path.parent_path();
    config_path += "/" + base + "_config.txt";
    int id = _pimpl->obj_system.ParseConfigFile(config_path.string().c_str());

    if(id >= 0)
    {
        interface->SetProjectIdForAllConstructors(id);
    }
    SetupObjectConstructors(interface);
    _pimpl->plugins.push_back(fullPluginPath + " - success");
    return true;
}

#endif

bool MetaObjectFactory::AbortCompilation()
{
	return _pimpl->obj_system.AbortCompilation();
}
bool MetaObjectFactory::CheckCompile()
{
	static boost::posix_time::ptime prevTime = boost::posix_time::microsec_clock::universal_time();
	boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
	boost::posix_time::time_duration delta = currentTime - prevTime;
	if (delta.total_milliseconds() < 10)
		return false;
	_pimpl->obj_system.GetFileChangeNotifier()->Update(float(delta.total_milliseconds()) / 1000.0f);
	return IsCurrentlyCompiling();
}
bool MetaObjectFactory::IsCurrentlyCompiling()
{
	return _pimpl->obj_system.GetIsCompiling();
}
bool MetaObjectFactory::IsCompileComplete()
{
	return _pimpl->obj_system.GetIsCompiledComplete();
}
bool MetaObjectFactory::SwapObjects()
{
	if (IsCompileComplete())
	{
		return _pimpl->obj_system.LoadCompiledModule();
	}
	return false;
}
void MetaObjectFactory::SetCompileCallback(std::function<void(const std::string, int)>& f)
{
       
}
std::shared_ptr<Connection> MetaObjectFactory::ConnectConstructorAdded(TypedSlot<void(void)>* slot)
{
    return _pimpl->on_constructor_added.Connect(slot);
}
