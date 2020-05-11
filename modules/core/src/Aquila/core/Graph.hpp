#pragma once
#include "IGraph.hpp"

#include <MetaObject/core/SystemTable.hpp>
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/object/detail/MetaObjectMacros.hpp>
#include <MetaObject/signals/detail/SignalMacros.hpp>
#include <MetaObject/signals/detail/SlotMacros.hpp>
#include <MetaObject/thread/ThreadHandle.hpp>

#include <boost/thread.hpp>

namespace aq
{

    class AQUILA_EXPORTS Graph : public IGraph
    {
      public:
        Graph();
        ~Graph() override;
        MO_DERIVE(Graph, IGraph)
            MO_SIGNAL(void, StartThreads)
            MO_SIGNAL(void, StopThreads)

            MO_SLOT(void, startThread)
            MO_SLOT(void, input_changed, nodes::INode*, mo::ISubscriber*)
            MO_SLOT(void, stopThread)
            MO_SLOT(void, pauseThread)
            MO_SLOT(void, resumeThread)

            MO_SLOT(void, node_updated, nodes::INode*)
            MO_SLOT(void, update)
            MO_SLOT(void, param_updated, mo::IMetaObject*, mo::IParam*)
            MO_SLOT(void, param_added, mo::IMetaObject*, mo::IParam*)

            MO_SLOT(void, run_continuously, bool)
            MO_SLOT(int, process)
        MO_END;

        std::vector<rcc::weak_ptr<aq::nodes::INode>> getTopLevelNodes() override;
        mo::IAsyncStreamPtr_t getStream() const override;
        void initCustom(bool firstInit) override;
        std::shared_ptr<mo::IParamServer> getParamServer() override;
        std::shared_ptr<mo::RelayManager> getRelayManager() override;
        rcc::shared_ptr<IUserController> getUserController() override;
        std::vector<rcc::shared_ptr<nodes::INode>> getNodes() const override;
        std::vector<rcc::shared_ptr<nodes::INode>> getAllNodes() const override;
        bool getDirty() const override;
        bool loadDocument(const std::string& document, const std::string& prefered_loader = "") override;
        std::vector<rcc::shared_ptr<nodes::INode>> addNode(const std::string& nodeName) override;
        void addNode(const rcc::shared_ptr<nodes::INode>& node) override;
        void addNodeNoInit(const rcc::shared_ptr<nodes::INode>& node);
        void addNodes(const std::vector<rcc::shared_ptr<nodes::INode>>& node) override;
        rcc::shared_ptr<nodes::INode> getNode(const std::string& nodeName) override;
        void removeNode(const nodes::INode* node) override;

        bool saveGraph(const std::string& filename) override;
        bool loadGraph(const std::string& filename) override;
        bool waitForSignal(const std::string& name, const boost::optional<std::chrono::milliseconds>& timeout) override;

        void addVariableSink(IVariableSink* sink) override;
        void removeVariableSink(IVariableSink* sink) override;

      protected:
        friend class IGraph;
        void addChildNode(rcc::shared_ptr<nodes::INode> node) override;

        IObjectContainer* getObjectContainer(mo::TypeInfo) const override;
        void setObjectContainer(mo::TypeInfo, IObjectContainer::Ptr_t&&) override;

        std::map<mo::TypeInfo, std::unique_ptr<IObjectContainer>> m_objects;
        std::shared_ptr<mo::IParamServer> m_param_server;
        std::shared_ptr<mo::RelayManager> m_relay_manager;
        std::mutex m_nodes_mtx;
        volatile bool m_dirty_flag = true;
        std::vector<IVariableSink*> m_variable_sinks;
        // These are threads for attempted connections
        std::vector<boost::thread*> m_connection_threads;
        std::vector<rcc::shared_ptr<nodes::INode>> m_top_level_nodes;
        std::vector<rcc::weak_ptr<nodes::INode>> m_child_nodes;
        rcc::shared_ptr<IUserController> m_window_callback_handler;
    };
} // namespace aq
