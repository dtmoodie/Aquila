#ifndef AQ_NODES_INODE_HPP
#define AQ_NODES_INODE_HPP
#include <Aquila/core/Algorithm.hpp>

#include <MetaObject/params.hpp>
#include <MetaObject/signals/detail/SlotMacros.hpp>
namespace aq
{
    class IGraph;
    namespace nodes
    {
        template <class T>
        struct TNodeInterfaceHelper;
        struct NodeInfo;
        class AQUILA_EXPORTS INode : virtual public TInterface<INode, Algorithm>
        {
          public:
            static std::vector<std::string> listConstructableNodes(const std::string& filter = "");

            template <class T>
            using InterfaceHelper = TNodeInterfaceHelper<T>;

            using InterfaceInfo = NodeInfo;
            using Ptr = rcc::shared_ptr<INode>;
            using WeakPtr = rcc::weak_ptr<INode>;
            using VecPtr = std::vector<Ptr>;

            MO_DERIVE(INode, Algorithm)
                MO_SLOT(void, reset)
                MO_SIGNAL(void, node_updated, INode*)
                MO_SIGNAL(void, input_changed, INode*, mo::ISubscriber*)
            MO_END;

            // operators on parents
            virtual void addParent(WeakPtr parent) = 0;
            virtual std::vector<WeakPtr> getParents() const = 0;

            // operators on children
            virtual void addChild(Ptr child) = 0;
            virtual Ptr getChild(const std::string& treeName) = 0;
            virtual Ptr getChild(const int& index) = 0;
            virtual VecPtr getChildren() = 0;
            virtual void removeChild(const std::string& name) = 0;
            virtual void removeChild(const INode* node) = 0;
            virtual void removeChild(int idx) = 0;

            // operators on graph
            virtual void setGraph(rcc::weak_ptr<IGraph> stream) = 0;
            virtual rcc::shared_ptr<IGraph> getGraph() = 0;

            // naming
            virtual void setUniqueId(int id) = 0;
            virtual std::string getName() const = 0;
            virtual void setName(const std::string& name) = 0;

            // initialization
            virtual void nodeInit(bool firstInit) = 0;

            // state tracking
            virtual bool getModified() const = 0;

          protected:
            virtual void setModified(bool val) = 0;
        };
    } // namespace nodes
} // namespace aq

#endif // AQ_NODES_INODE_HPP