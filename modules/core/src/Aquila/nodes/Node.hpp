#pragma once

/*
 *  Nodes are the base processing object of this library.  Nodes can encapsulate all aspects of image processing from
 *  a single operation to a parallel set of intercommunicating processing stacks.  Nodes achieve this through one or more
 *  of the following concepts:
 *
 *  1) Processing node - Input image -> output image
 *  2) Function node - publishes a boost::function object to be used by sibling nodes
 *  3) Object node - publishes an object to be used by sibling nodes
 *  4) Serial collection nodes - Input image gets processed by all child nodes in series
 *  5) Parallel collection nodes - Input image gets processed by all child nodes in parallel. One thread per node
 *
 *  Nodes should be organized in a tree structure.  Each node will be accessible by name from the top of the tree via /parent/...../treeName where
 *  treeName is the unique name associated with that node.  The parameters of that node can be accessed via /parent/...../treeName:name.
 *  Nodes should be iterable by their parents by insertion order.  They should be accessible by sibling nodes.
 *
 *  Parameters:
 *  - Four main types of parameters, input, output, control and status.
 *  -- Input parameters should be defined in the expecting node by their internal name, datatype and the name of the output assigned to this input
 *  -- Output parameters should be defined by their datatype, memory location and
 *  - Designs:
 *  -- Could have a vector of parameter objects for each type.  The parameter object would contain the tree name of the parameter,
 *      the parameter type, and a pointer to the parameter
 *  -- Other considerations include using a boost::shared_ptr to the parameter, in which case constructing node and any other node that uses the parameter would share access.
 *      this has the advantage that the parameters don't need to be updated when an object swap occurs, since they aren't deleted.
 *      This would be nice for complex parameter objects, but it has the downside of functors not being updated correctly, which isn't such a big deal because the
 *      developer should just update functors accordingly in the init(bool) function.
 *
*/

// In library includes
#include "Aquila/core/detail/Export.hpp"
#include "Aquila/core/Algorithm.hpp"
//#include "Aquila/core/IDataStream.hpp"
//#include "Aquila/utilities/cuda/CudaUtils.hpp"
//#include <Aquila/IO/serialize.hpp>

// RCC includes
#include <IObject.h>
#include <IObjectInfo.h>
#include <RuntimeObjectSystem/ObjectInterfacePerModule.h>
#include <RuntimeObjectSystem/RuntimeLinkLibrary.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>

// Dependent in house libraries
#include <MetaObject/object/MetaObject.hpp>

// Dependent 3rd party libraries
#include <opencv2/core/cuda.hpp>
#include <Aquila/rcc/external_includes/cv_core.hpp>
#include <Aquila/rcc/external_includes/cv_highgui.hpp>

#include <string>

#define SCOPED_PROFILE_NODE mo::scoped_profile COMBINE(scoped_profile, __LINE__)((this->getTreeName() + "::" + __FUNCTION__), nullptr, nullptr, cudaStream());
#define LOG_NODE(severity) BOOST_LOG_TRIVIAL(severity) << "[" << this->getTreeName() << "::" << __FUNCTION__ <<  "] - "

namespace mo{
    class IVariableManager;
    typedef std::shared_ptr<IVariableManager> IVariableManagerPtr;
}
namespace cereal{
    class JSONInputArchive;
    class JSONOutputArchive;
}

namespace aq{
namespace nodes{
    class Node;
    class NodeImpl;
}
class IDataStream;
class DataStream;
class NodeFactory;
AQUILA_EXPORTS bool DeSerialize(cereal::JSONInputArchive& ar, aq::nodes::Node* obj);
AQUILA_EXPORTS bool Serialize(cereal::JSONOutputArchive& ar, const aq::nodes::Node* obj);
namespace nodes{

    struct NodeInfo;

    class AQUILA_EXPORTS Node:
            public TInterface<Node, Algorithm>
    {
    public:
        typedef NodeInfo InterfaceInfo;
        typedef rcc::shared_ptr<Node> Ptr;
        typedef rcc::weak_ptr<Node>   WeakPtr;
        typedef std::vector<Ptr> VecPtr;
        static std::vector<std::string> listConstructableNodes(const std::string& filter = "");
        Node();
        virtual bool                    process();

        virtual void                    addParent(Node *parent);

        std::vector<WeakPtr>            getParents() const;

        virtual bool                    connectInput(Ptr output_node,
                                                     const std::string& output_name,
                                                     const std::string& input_name,
                                                     mo::ParamType type = mo::StreamBuffer_e);
        virtual bool                    connectInput(Ptr output_node,
                                                     mo::IParam* output_param,
                                                     mo::InputParam* input_param,
                                                     mo::ParamType type = mo::StreamBuffer_e);

        virtual Ptr                     addChild(Node* child);
        virtual Ptr                     addChild(Node::Ptr child);
        virtual void addComponent(const rcc::weak_ptr<Algorithm>& component);
        virtual Ptr                     getChild(const std::string& treeName);
        virtual Ptr                     getChild(const int& index);
        virtual VecPtr                  getChildren();

        virtual void                    removeChild(const std::string& name);
        virtual void                    removeChild(Ptr node);
        virtual void                    removeChild(Node* node);
        virtual void                    removeChild(WeakPtr node);
        virtual void                    removeChild(int idx);

        virtual void                    swapChildren(int idx1, int idx2);
        virtual void                    swapChildren(const std::string& name1, const std::string& name2);
        virtual void                    swapChildren(Node::Ptr child1, Node::Ptr child2);

        virtual void                    setDataStream(IDataStream* stream);
        virtual IDataStream*            getDataStream();
        virtual mo::IVariableManagerPtr getVariableManager();

        void                            setUniqueId(int id);
        std::string                     getTreeName();
        std::string                     getTreeName() const;
        void                            setTreeName(const std::string& name);

        virtual void                    Init(bool firstInit);
        virtual void                    nodeInit(bool firstInit);
        virtual void                    postSerializeInit();

        virtual void                    Serialize(ISimpleSerializer *pSerializer);
        inline cv::cuda::Stream&        stream(){ CV_Assert(_ctx.get()); return _ctx.get()->getStream();}
        inline cudaStream_t             cudaStream(){CV_Assert(_ctx.get()); return _ctx.get()->getCudaStream();}

        InputState                      checkInputs();

        MO_DERIVE(Node, Algorithm)
            MO_SLOT(void, reset)
            MO_SIGNAL(void, node_updated, Node*)
            MO_SIGNAL(void, input_changed, Node*, mo::InputParam*)
        MO_END
        bool getModified() const;
    protected:
        friend class NodeFactory;
        friend class IDataStream;
        friend class aq::DataStream;

        virtual std::vector<Node*>  getNodesInScope();
        virtual Node *              getNodeInScope(const std::string& name);
        virtual void                getNodesInScope(std::vector<Node*>& nodes);
        virtual mo::IParam*         addParameter(std::shared_ptr<mo::IParam> param);
        virtual mo::IParam*         addParameter(mo::IParam* param);

        friend bool aq::DeSerialize(cereal::JSONInputArchive& ar, Node* obj);
        friend bool aq::Serialize(cereal::JSONOutputArchive& ar, const aq::nodes::Node* obj);

        virtual void onParamUpdate(mo::IParam*, mo::Context*, mo::OptionalTime_t, size_t, const std::shared_ptr<mo::ICoordinateSystem>&, mo::UpdateFlags);
        // Current timestamp of the frame that this node is processing / processed last

        // The variable manager is one object shared within a processing graph
        // that has knowledge of all inputs and outputs within the graph
        // It handles creating buffers, setting up contexts and all connecting nodes
        mo::IVariableManagerPtr                         _variable_manager;
        bool                                            _modified;
        // The children of a node are all nodes accepting inputs
        // from this node
        VecPtr                                          _children;
        rcc::weak_ptr<IDataStream>                      _data_stream;
        int _unique_id;
        std::vector<WeakPtr>                            _parents;
        std::string                                     name;
    private:
        std::shared_ptr<NodeImpl>                       _pimpl_node;
    };
} // namespace nodes
} // namespace aq
