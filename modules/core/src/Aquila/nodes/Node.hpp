#pragma once

/*
 *  Nodes are the base processing object of this library.  Nodes can encapsulate all aspects of image processing from
 *  a single operation to a parallel set of intercommunicating processing stacks.  Nodes achieve this through one or
 * more
 *  of the following concepts:
 *
 *  1) Processing node - Input image -> output image
 *  2) Function node - publishes a boost::function object to be used by sibling nodes
 *  3) Object node - publishes an object to be used by sibling nodes
 *  4) Serial collection nodes - Input image gets processed by all child nodes in series
 *  5) Parallel collection nodes - Input image gets processed by all child nodes in parallel. One thread per node
 *
 *  Nodes should be organized in a tree structure.  Each node will be accessible by name from the top of the tree via
 * /parent/...../treeName where
 *  treeName is the unique name associated with that node.  The parameters of that node can be accessed via
 * /parent/...../treeName:name.
 *  Nodes should be iterable by their parents by insertion order.  They should be accessible by sibling nodes.
 *
 *  Parameters:
 *  - Four main types of parameters, input, output, control and status.
 *  -- Input parameters should be defined in the expecting node by their internal name, datatype and the name of the
 * output assigned to this input
 *  -- Output parameters should be defined by their datatype, memory location and
 *  - Designs:
 *  -- Could have a vector of parameter objects for each type.  The parameter object would contain the tree name of the
 * parameter,
 *      the parameter type, and a pointer to the parameter
 *  -- Other considerations include using a boost::shared_ptr to the parameter, in which case constructing node and any
 * other node that uses the parameter would share access.
 *      this has the advantage that the parameters don't need to be updated when an object swap occurs, since they
 * aren't deleted.
 *      This would be nice for complex parameter objects, but it has the downside of functors not being updated
 * correctly, which isn't such a big deal because the
 *      developer should just update functors accordingly in the init(bool) function.
 *
 */

// In library includes
#include "Aquila/core/Algorithm.hpp"
//#include "Aquila/detail/export.hpp"
#include "Aquila/nodes/INode.hpp"

// Dependent in house libraries
#include <MetaObject/object/MetaObject.hpp>
#include <MetaObject/params/ParamMacros.hpp>
#include <MetaObject/signals/detail/SignalMacros.hpp>
#include <MetaObject/signals/detail/SlotMacros.hpp>

// RCC includes
#include <RuntimeObjectSystem/IObject.h>
#include <RuntimeObjectSystem/IObjectInfo.h>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>

#include <string>

#include "RuntimeObjectSystem/RuntimeLinkLibrary.h"

namespace mo
{
    class IParamServer;
    using IParamServerPtr = std::shared_ptr<IParamServer>;
} // namespace mo

namespace aq
{
    class IGraph;
    class Graph;

    namespace nodes
    {
        struct NodeInfo;

        class AQUILA_EXPORTS Node : virtual public INode
        {
          public:
            MO_DERIVE(Node, INode)
            MO_END;

            Node();

            ~Node() override;

            bool process(mo::IAsyncStream&) override;

            void addParent(WeakPtr parent) override;
            std::vector<WeakPtr> getParents() const override;

            bool connectInput(const std::string& input_name,
                              mo::IMetaObject* output_object,
                              const std::string& output_name,
                              mo::BufferFlags type = mo::BufferFlags::BLOCKING_STREAM_BUFFER) override;

            bool connectInput(mo::ISubscriber* input,
                              IMetaObject* output_object,
                              mo::IPublisher* output_param,
                              mo::BufferFlags type = mo::BufferFlags::BLOCKING_STREAM_BUFFER) override;

            void addComponent(const rcc::weak_ptr<IAlgorithm>& component) override;

            void addChild(Ptr child) override;

            Ptr getChild(const std::string& treeName) override;
            Ptr getChild(const int& index) override;
            VecPtr getChildren() override;

            void removeChild(const std::string& name) override;
            void removeChild(const INode* node) override;
            void removeChild(int idx) override;

            void setGraph(rcc::weak_ptr<IGraph> graph) override;
            rcc::shared_ptr<IGraph> getGraph() override;
            mo::IParamServerPtr getParamServer();

            void setUniqueId(int id) override;
            std::string getName() const override;
            void setName(const std::string& name) override;

            void Init(bool firstInit) override;
            void nodeInit(bool firstInit) override;
            void postSerializeInit() override;

            void Serialize(ISimpleSerializer* pSerializer) override;

            bool getModified() const override;
            void addParam(std::shared_ptr<mo::IParam> param) override;
            void addParam(mo::IParam& param) override;

          protected:
            bool processChildren(mo::IAsyncStream&);

            void setModified(bool val = true) override;
            void onParamUpdate(const mo::IParam&, mo::Header, mo::UpdateFlags, mo::IAsyncStream&) override;

          private:
            std::vector<WeakPtr> m_parents;
            VecPtr m_children;
            rcc::weak_ptr<IGraph> m_graph;

            bool m_modified = false;
            int m_unique_id;
            mutable std::string m_name;
            long long m_throw_count = 0;
            bool m_disable_due_to_errors = false;
            std::string m_tree_name;
            long long m_iterations_since_execution = 0;
            const char* m_last_execution_failure_reason = nullptr;
        };
    } // namespace nodes
} // namespace aq
