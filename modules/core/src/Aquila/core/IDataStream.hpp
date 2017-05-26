#pragma once
#include "Aquila/core/detail/Export.hpp"
#include <MetaObject/object/IMetaObject.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <type_traits>
#include <vector>
#include <memory>
#include <map>

struct ISingleton;
template<class T> struct Singleton;
template<class T> struct IObjectSingleton;

namespace aq
{
    namespace Nodes
    {
        class Node;
        class IFrameGrabber;
    }
    class IRenderEngine;
    class IParameterBuffer;
    class IVariableSink;
    class WindowCallbackHandler;

    class AQUILA_EXPORTS IDataStream: public TInterface<IDataStream, mo::IMetaObject>
    {
    public:
        typedef rcc::shared_ptr<IDataStream> Ptr;
        typedef std::map<std::string, std::string> VariableMap;
        static Ptr create(const std::string& document = "", const std::string& preferred_frame_grabber = "");
        /*!
         * \brief Loads a data stream configuration file from disk
         * \param config_file is the configuration file from disk, currently supports json
         * \param vm variable replacement map, all variables with name==key will be replace with vm[key]
         * \param sm string replacement map, all strings found in sm with the format ${key} will be replaced with sm[key]
         * \return vector of data streams loaded from file
         */
        static std::vector<Ptr> load(const std::string& config_file, VariableMap& vm, VariableMap& sm);
        static std::vector<Ptr> load(const std::string& config_file);
        static void save(const std::string& config_file, rcc::shared_ptr<IDataStream>& stream);
        static void save(const std::string& config_file, std::vector<rcc::shared_ptr<IDataStream>>& streams);
        static void save(const std::string& config_file, std::vector<rcc::shared_ptr<IDataStream>>& streams,
                         const VariableMap& vm, const VariableMap& sm);
        static bool canLoadDocument(const std::string& document);

        virtual std::vector<rcc::weak_ptr<Nodes::Node>> getTopLevelNodes() = 0;

        // Handles actual rendering of data.  Use for adding extra objects to the scene
        virtual mo::RelayManager*                           getRelayManager() = 0;
        virtual std::shared_ptr<mo::IVariableManager>       getVariableManager() = 0;
        virtual rcc::weak_ptr<WindowCallbackHandler>        getWindowCallbackManager() = 0;
        virtual IParameterBuffer*                           getParameterBuffer() = 0;
        virtual std::vector<rcc::shared_ptr<Nodes::Node>>   getNodes() const = 0;
        virtual std::vector<rcc::shared_ptr<Nodes::Node>>   getAllNodes() const = 0;
        virtual bool loadDocument(const std::string& document, const std::string& prefered_loader = "") = 0;

        virtual std::vector<rcc::shared_ptr<Nodes::Node>> addNode(const std::string& nodeName) = 0;
        virtual void addNode(rcc::shared_ptr<Nodes::Node> node) = 0;
        virtual void addNodes(std::vector<rcc::shared_ptr<Nodes::Node>> node) = 0;
        virtual void removeNode(rcc::shared_ptr<Nodes::Node> node) = 0;
        virtual void removeNode(Nodes::Node* node) = 0;
        virtual Nodes::Node* getNode(const std::string& nodeName) = 0;

        virtual void startThread() = 0;
        virtual void stopThread() = 0;
        virtual void pauseThread() = 0;
        virtual void resumeThread() = 0;
        virtual int process() = 0;

        virtual void addVariableSink(IVariableSink* sink) = 0;
        virtual void removeVariableSink(IVariableSink* sink) = 0;

        virtual bool saveStream(const std::string& filename) = 0;
        virtual bool loadStream(const std::string& filename) = 0;

        // Get or create singleton for this datastream
        template<typename T> T* getSingleton(){
            auto& ptr_ref = getSingleton(mo::TypeInfo(typeid(T)));
            if(ptr_ref){
                return static_cast<Singleton<T>*>(ptr_ref.get())->ptr;
            }else{
                ptr_ref.reset(new Singleton<T>(new T()));
            }
        }

        template<typename T> rcc::weak_ptr<T> getIObjectSingleton(){
            auto& ptr_ref = getIObjectSingleton(mo::TypeInfo(typeid(T)));
            if(ptr_ref){
                return static_cast<IObjectSingleton<T>*>(ptr_ref.get())->ptr;
            }else{
                auto ptr = new IObjectSingleton<T>(T::create());
                ptr_ref.reset(ptr);
                return ptr->ptr;
            }
        }

        // Transfers ownership to the datastream
        template<typename T> T* setSingleton(typename std::enable_if<!std::is_base_of<IObject, T>::value, T>* singleton){
            auto& ptr_ref = getSingleton(mo::TypeInfo(typeid(T)));
            ptr_ref.reset(new Singleton<T>(singleton));
            return singleton;
        }
        template<typename T> rcc::weak_ptr<T> setSingleton(typename std::enable_if<std::is_base_of<IObject, T>::value, T>* singleton){
            auto& ptr_ref = getIObjectSingleton(mo::TypeInfo(typeid(T)));
            auto sig_ptr = new IObjectSingleton<T>(singleton);
            ptr_ref.reset(sig_ptr);
            return sig_ptr->ptr;;
        }
    protected:
        friend class Nodes::Node;
        virtual void addChildNode(rcc::shared_ptr<Nodes::Node> node) = 0;
        virtual void removeChildNode(rcc::shared_ptr<Nodes::Node> node) = 0;
        virtual std::unique_ptr<ISingleton>& getSingleton(mo::TypeInfo type) = 0;
        virtual std::unique_ptr<ISingleton>& getIObjectSingleton(mo::TypeInfo type) = 0;
    };
}
