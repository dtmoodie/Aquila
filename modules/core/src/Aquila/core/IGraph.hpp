#pragma once
#include "Aquila/core/detail/Export.hpp"
#include <MetaObject/detail/TypeInfo.hpp>
#include <MetaObject/object/MetaObject.hpp>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <map>
#include <memory>
#include <type_traits>
#include <vector>

namespace mo
{
    class IParamServer;
    struct ISingletonContainer;
    template <class T>
    struct TSingletonContainer;
    template <class T>
    struct TIObjectSingletonContainer;
} // namespace mo

namespace aq
{
    class JSONInputArchive;
    class JSONOutputArchive;
    namespace nodes
    {
        class INode;
        class Node;
        class IFrameGrabber;
    } // namespace nodes
    class IRenderEngine;
    class IVariableSink;
    class IUserController;

    class AQUILA_EXPORTS IGraph : public TInterface<IGraph, mo::MetaObject>, virtual protected mo::IObjectTable
    {
      public:
        using Ptr_t = rcc::shared_ptr<IGraph>;
        using VariableMap_t = std::map<std::string, std::string>;
        static Ptr_t create(const std::string& document = "", const std::string& preferred_frame_grabber = "");

        static bool canLoadPath(const std::string& document);

        virtual std::vector<rcc::weak_ptr<nodes::INode>> getTopLevelNodes() = 0;

        // Handles actual rendering of data.  Use for adding extra objects to the scene
        virtual std::shared_ptr<mo::RelayManager> getRelayManager() = 0;
        virtual rcc::shared_ptr<IUserController> getUserController() = 0;
        virtual std::shared_ptr<mo::IParamServer> getParamServer() = 0;
        virtual std::vector<rcc::shared_ptr<nodes::INode>> getNodes() const = 0;
        virtual std::vector<rcc::shared_ptr<nodes::INode>> getAllNodes() const = 0;
        virtual bool getDirty() const = 0;
        virtual bool loadDocument(const std::string& document, const std::string& prefered_loader = "") = 0;

        virtual std::vector<rcc::shared_ptr<nodes::INode>> addNode(const std::string& nodeName) = 0;
        virtual void addNode(const rcc::shared_ptr<nodes::INode>& node) = 0;
        virtual void addNodes(const std::vector<rcc::shared_ptr<nodes::INode>>& node) = 0;
        virtual void removeNode(const nodes::INode* node) = 0;
        virtual rcc::shared_ptr<nodes::INode> getNode(const std::string& nodeName) = 0;

        virtual void startThread() = 0;
        virtual void stopThread() = 0;
        virtual void pauseThread() = 0;
        virtual void resumeThread() = 0;
        virtual int process() = 0;

        virtual void addVariableSink(IVariableSink* sink) = 0;
        virtual void removeVariableSink(IVariableSink* sink) = 0;

        virtual bool saveGraph(const std::string& filename) = 0;
        virtual bool loadGraph(const std::string& filename) = 0;

        virtual bool waitForSignal(const std::string& name,
                                   const boost::optional<std::chrono::milliseconds>& timeout =
                                       boost::optional<std::chrono::milliseconds>()) = 0;

        template <typename T, class U = T>
        mo::SharedPtrType<T> getObject()
        {
            return IObjectTable::getObject<T, U>();
        }
        virtual void addChildNode(rcc::shared_ptr<nodes::INode> node) = 0;
    };
} // namespace aq
