#include "Aquila/nodes/NodeFactory.hpp"
#include "Aquila/core/IGraph.hpp"
#include "Aquila/nodes/Node.hpp"
#include "Aquila/nodes/NodeInfo.hpp"

#include <MetaObject/object/MetaObjectFactory.hpp>
#include <MetaObject/params/ISubscriber.hpp>

#include "RuntimeCompiler/AUArray.h"
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
using namespace aq;

NodeFactory* NodeFactory::Instance()
{
    static NodeFactory instance;
    return &instance;
}

NodeFactory::NodeFactory()
{
}

NodeFactory::~NodeFactory()
{
}

// I don't think this is needed anymore
void NodeFactory::onConstructorsAdded()
{
    /*auto constructors = mo::MetaObjectFactory::instance()->getConstructors(aq::nodes::Node::getHash());
    std::vector<nodes::Node*> newNodes;
    for (size_t i = 0; i < constructors.size(); ++i)
    {
        size_t numObjects = constructors[i]->GetNumberConstructedObjects();
        for (size_t j = 0; j < numObjects; ++j)
        {
            auto ptr = constructors[i]->GetConstructedObject(j);
            if (ptr)
            {
                auto iptr = ptr->GetInterface(aq::nodes::Node::getHash());
                if (iptr)
                {
                    ptr = static_cast<aq::nodes::INode*>(iptr);
                    auto nodePtr = dynamic_cast<nodes::Node*>(ptr);
                    newNodes.push_back(nodePtr);
                }
            }
        }
    }
    for (size_t i = 0; i < newNodes.size(); ++i)
    {
        auto parameters = newNodes[i]->getDisplayParams();
        for (size_t j = 0; j < parameters.size(); ++j)
        {
            if (parameters[j]->checkFlags(mo::ParamFlags::kINPUT))
            {
                auto inputParam = dynamic_cast<mo::InputParam*>(parameters[j]);
                if (inputParam)
                    inputParam->setInput();
            }
        }
    }*/
}

rcc::shared_ptr<nodes::INode> NodeFactory::addNode(const std::string& node_name)
{
    auto pConstructor = mo::MetaObjectFactory::instance()->getConstructor(node_name.c_str());

    if (pConstructor && pConstructor->GetInterfaceId() == nodes::Node::getHash())
    {
        IObject* pObj = pConstructor->Construct();
        IObject* interface = static_cast<IObject*>(pObj->GetInterface(nodes::Node::getHash()));

        if (interface)
        {
            try
            {
                interface->Init(true);
            }
            catch (cv::Exception& e)
            {
                MO_LOG(error, "Failed to initialize node {} due to : {}", node_name, e.what());
                return rcc::shared_ptr<nodes::Node>();
            }
            catch (...)
            {
                MO_LOG(error, "Failed to initialize node {}", node_name);
                return rcc::shared_ptr<nodes::Node>();
            }
            rcc::shared_ptr<nodes::INode> out(*interface);
            m_nodes.push_back(out);
            return out;
        }
        else
        {
            MO_LOG(warn, "[ NodeManager ] {} not a node", node_name);
            // Input nodename is a compatible object but it is not a node
            return rcc::shared_ptr<nodes::Node>();
        }
    }
    else
    {
        MO_LOG(warn, "[ NodeManager ] {} not a valid node name", node_name);
        return rcc::shared_ptr<nodes::Node>();
    }

    return rcc::shared_ptr<nodes::Node>();
}
// WIP needs to be tested for complex dependency trees
std::vector<rcc::shared_ptr<nodes::INode>> NodeFactory::addNode(const std::string& nodeName, IGraph* parentStream)
{
    IObjectConstructor* pConstructor = mo::MetaObjectFactory::instance()->getConstructor(nodeName.c_str());
    std::vector<rcc::shared_ptr<nodes::INode>> constructed_nodes;
    if (pConstructor && pConstructor->GetInterfaceId() == nodes::Node::getHash())
    {
        auto obj_info = pConstructor->GetObjectInfo();
        auto node_info = dynamic_cast<const nodes::NodeInfo*>(obj_info);
        auto parental_deps = node_info->getParentalDependencies();
        // Since a data stream is selected and by definition a parental dependency must be in the direct parental path,
        // we build all parent dependencies
        rcc::shared_ptr<nodes::Node> parent_node;
        for (auto& parent_dep : parental_deps)
        {
            if (parent_dep.size())
            {
                if (parent_node)
                {
                    auto parent_nodes = addNode(parent_dep[0], parent_node.get());
                    constructed_nodes.insert(constructed_nodes.end(), parent_nodes.begin(), parent_nodes.end());
                    parent_node = parent_nodes.back();
                }
                else
                {
                    auto parent_nodes = addNode(parent_dep[0], parentStream);
                    constructed_nodes.insert(constructed_nodes.end(), parent_nodes.begin(), parent_nodes.end());
                    parent_node = parent_nodes.back();
                }
            }
        }
        auto non_parent_deps = node_info->getNonParentalDependencies();
        auto existing_nodes = parentStream->getNodes();
        for (auto& non_parent_dep : non_parent_deps)
        {
            bool found = false;
            for (auto& existing_node : existing_nodes)
            {
                for (auto& dep : non_parent_dep)
                {
                    if (existing_node->GetTypeName() == dep)
                    {
                        found = true;
                        break;
                    }
                }
            }
            // No qualified parental dependency was found, add first best candidate
            if (!found)
            {
                auto added_nodes = addNode(non_parent_dep[0], parentStream);
                constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
            }
        }
        auto dependent_variable_nodes = node_info->checkDependentVariables(parentStream->getParamServer().get());
        for (auto& dependent_variable_node : dependent_variable_nodes)
        {
            auto added_nodes = addNode(dependent_variable_node, parentStream);
            constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
        }
        // All dependencies have been handled, construct node
        auto iobject_ptr = pConstructor->Construct();
        rcc::shared_ptr<aq::nodes::INode> node_ptr(iobject_ptr);
        if (node_ptr)
        {
            node_ptr->Init(true);
            m_nodes.push_back(node_ptr);
            constructed_nodes.push_back(node_ptr);
            if (parent_node)
            {
                parent_node->addChild(node_ptr);
            }
            else
            {
                parentStream->addNode(node_ptr);
            }
        }
    }
    return constructed_nodes;
}

// recursively checks if a node exists in the parent hierarchy
bool check_parent_exists(nodes::INode* node, const std::string& name)
{
    if (node->GetTypeName() == name)
        return true;
    /*if (auto parent = node->getParent())
        return check_parent_exists(parent, name);*/
    return false;
}

std::vector<rcc::shared_ptr<nodes::INode>> NodeFactory::addNode(const std::string& nodeName, nodes::INode* parent_node)
{
    IObjectConstructor* pConstructor = mo::MetaObjectFactory::instance()->getConstructor(nodeName.c_str());
    std::vector<rcc::shared_ptr<nodes::INode>> constructed_nodes;
    if (pConstructor && pConstructor->GetInterfaceId() == nodes::Node::getHash())
    {
        auto obj_info = pConstructor->GetObjectInfo();
        auto node_info = dynamic_cast<const nodes::NodeInfo*>(obj_info);
        auto parental_deps = node_info->getParentalDependencies();
        rcc::shared_ptr<nodes::INode> parent_node;
        for (auto& parent_dep : parental_deps)
        {
            if (parent_dep.size())
            {
                // For each node already in this tree, search for any of the allowed parental dependencies
                bool found = false;
                for (auto& dep : parent_dep)
                {
                    if (check_parent_exists(parent_node, dep))
                    {
                        found = true;
                        break;
                    }
                }
                if (!found)
                {
                    if (parent_node)
                    {
                        auto parent_nodes = addNode(parent_dep[0], parent_node.get());
                        constructed_nodes.insert(constructed_nodes.end(), parent_nodes.begin(), parent_nodes.end());
                    }
                    else
                    {
                        auto parent_nodes = addNode(parent_dep[0], parent_node);
                        constructed_nodes.insert(constructed_nodes.end(), parent_nodes.begin(), parent_nodes.end());
                        parent_node = parent_nodes.back();
                    }
                }
            }
        }
        auto non_parent_deps = node_info->getNonParentalDependencies();
        auto graph = parent_node->getGraph();
        if (graph)
        {
            auto existing_nodes = graph->getNodes();
            for (auto& non_parent_dep : non_parent_deps)
            {
                bool found = false;
                for (auto& existing_node : existing_nodes)
                {
                    for (auto& dep : non_parent_dep)
                    {
                        if (existing_node->GetTypeName() == dep)
                        {
                            found = true;
                            break;
                        }
                    }
                }
                // No qualified parental dependency was found, add first best candidate
                if (!found)
                {
                    auto added_nodes = addNode(non_parent_dep[0], parent_node);
                    constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
                }
            }
            auto dependent_variable_nodes = node_info->checkDependentVariables(graph->getParamServer().get());
            for (auto& dependent_variable_node : dependent_variable_nodes)
            {
                auto added_nodes = addNode(dependent_variable_node, parent_node);
                constructed_nodes.insert(constructed_nodes.end(), added_nodes.begin(), added_nodes.end());
            }
        }

        rcc::shared_ptr<nodes::INode> node(pConstructor->Construct());
        node->Init(true);
        parent_node->addChild(node);
        constructed_nodes.push_back(node);
    }
    return constructed_nodes;
}

bool NodeFactory::removeNode(const std::string& nodeName)
{
    return false;
}

std::string NodeFactory::getNodeFile(const ObjectId& id)
{
    auto constructors = mo::MetaObjectFactory::instance()->getConstructors();
    if (constructors.size() > id.m_ConstructorId)
    {
        return std::string(constructors[id.m_ConstructorId]->GetFileName());
    }
    return std::string();
}

bool NodeFactory::removeNode(ObjectId oid)
{
    return false;
}

void NodeFactory::RegisterNodeInfo(const char* nodeName, std::vector<char const*>& nodeInfo)
{
    m_nodeInfoMap[nodeName] = nodeInfo;
}

const nodes::NodeInfo* NodeFactory::getNodeInfo(std::string& nodeName)
{
    auto constructor = mo::MetaObjectFactory::instance()->getConstructor(nodeName.c_str());
    if (constructor)
    {
        auto obj_info = constructor->GetObjectInfo();
        if (obj_info)
        {
            if (obj_info->GetInterfaceId() == nodes::Node::getHash())
            {
                auto node_info = dynamic_cast<const aq::nodes::NodeInfo*>(obj_info);
                if (node_info)
                {
                    return node_info;
                }
            }
        }
    }
    return nullptr;
}

void NodeFactory::SaveTree(const std::string& fileName)
{
}

void NodeFactory::onNodeRecompile(nodes::Node* node)
{
}

nodes::INode* NodeFactory::getNode(const ObjectId& id)
{
    auto constructors = mo::MetaObjectFactory::instance()->getConstructors();
    if (!id.IsValid())
        return nullptr;
    if (id.m_ConstructorId >= constructors.size())
        return nullptr;
    if (id.m_PerTypeId >= constructors[id.m_ConstructorId]->GetNumberConstructedObjects())
        return nullptr;
    rcc::shared_ptr<nodes::INode> pObj = constructors[id.m_ConstructorId]->GetConstructedObject(id.m_PerTypeId);
    if (!pObj)
        return nullptr;
    auto node = pObj.DynamicCast<nodes::INode>();
    if (!node)
    {
        return nullptr;
    }
    return node.get();
}

nodes::INode* NodeFactory::getNode(const std::string& treeName)
{
    for (size_t i = 0; i < m_nodes.size(); ++i)
    {
        auto shared = m_nodes[i].lock();
        if (shared)
        {
            if (shared->getName() == treeName)
            {
                return shared.get();
            }
        }
    }
    return nullptr;
}

void NodeFactory::UpdateTreeName(nodes::Node* node, const std::string& prevTreeName)
{
}

void NodeFactory::GetSiblingNodes(const std::string& sourceNode, std::vector<nodes::Node*>& output)
{
}

void NodeFactory::printTreeHelper(std::stringstream& tree, int level, nodes::INode* node)
{

    for (int i = 0; i < level; ++i)
    {
        tree << "+";
    }
    tree << node->getName() << std::endl;
    auto children = node->getChildren();
    for (size_t i = 0; i < children.size(); ++i)
    {
        printTreeHelper(tree, level + 1, children[i].get());
    }
}

void NodeFactory::PrintNodeTree(std::string* ret)
{
    std::stringstream tree;
    std::vector<rcc::weak_ptr<nodes::INode>> parent_nodes;
    // First get the top level nodes for the tree
    for (size_t i = 0; i < m_nodes.size(); ++i)
    {
        auto shared = m_nodes[i].lock();
        if (shared)
        {
            auto parents = shared->getParents();
            parent_nodes.insert(parent_nodes.begin(), parents.begin(), parents.end());
        }
    }
    for (size_t i = 0; i < parent_nodes.size(); ++i)
    {
        auto shared = parent_nodes[i].lock();
        if (shared)
        {
            printTreeHelper(tree, 0, shared.get());
        }
    }
    if (ret)
    {
        *ret = tree.str();
    }
    else
    {
        std::cout << tree.str() << std::endl;
    }
}

nodes::Node* NodeFactory::GetParent(const std::string& sourceNode)
{

    return nullptr;
}
void NodeFactory::GetParentNodes(const std::string& sourceNode, std::vector<nodes::Node*>& output)
{
}

void NodeFactory::GetAccessibleNodes(const std::string& sourceNode, std::vector<nodes::Node*>& output)
{

    GetSiblingNodes(sourceNode, output);
    GetParentNodes(sourceNode, output);
}

std::vector<std::string> NodeFactory::GetConstructableNodes()
{
    auto constructors = mo::MetaObjectFactory::instance()->getConstructors();
    std::vector<std::string> output;
    for (size_t i = 0; i < constructors.size(); ++i)
    {
        if (constructors[i])
        {
            if (constructors[i]->GetInterfaceId() == nodes::Node::getHash())
                output.push_back(constructors[i]->GetName());
        }
        else
        {
            std::cout << "Null constructor idx " << i << std::endl;
        }
    }
    return output;
}

std::vector<std::string> NodeFactory::GetParametersOfType(std::function<bool(mo::TypeInfo)> selector)
{

    std::vector<std::string> parameters;
    for (size_t i = 0; i < m_nodes.size(); ++i)
    {
        auto shared = m_nodes[i].lock();
        if (shared)
        {
            auto node_params = shared->getParams();
            for (size_t j = 0; j < node_params.size(); ++j)
            {
                if (selector(node_params[j]->getTypeInfo()))
                    parameters.push_back(node_params[j]->getName());
            }
        }
    }
    return parameters;
}
