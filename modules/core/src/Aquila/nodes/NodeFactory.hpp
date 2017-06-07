#pragma once
#include <IObject.h>
#include "Aquila/core/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <opencv2/core/persistence.hpp>
#include <functional>
#include <vector>
#include <string>
#include <map>

namespace aq
{
    namespace Nodes
    {
        class Node;
        struct NodeInfo;
    }
    class IDataStream;

    class AQUILA_EXPORTS NodeFactory
    {
    public:
        static NodeFactory* Instance();

        void RegisterNodeInfo(const char* nodeName, std::vector<char const*>& nodeInfo);

        Nodes::NodeInfo* getNodeInfo(std::string& nodeName);

        rcc::shared_ptr<Nodes::Node> addNode(const std::string& nodeName);

        // Adds a node by name to the data stream or the parent node.
        std::vector<rcc::shared_ptr<Nodes::Node>> addNode(const std::string& nodeName, IDataStream* parentStream);
        std::vector<rcc::shared_ptr<Nodes::Node>> addNode(const std::string& nodeName, Nodes::Node* parentNode);

        void PrintNodeTree(std::string* ret = nullptr);
        void SaveTree(const std::string& fileName);
        std::string getNodeFile(const ObjectId& id);

        Nodes::Node* getNode(const ObjectId& id);
        Nodes::Node* getNode(const std::string& treeName);
        bool removeNode(const std::string& nodeName);
        bool removeNode(ObjectId oid);

        void UpdateTreeName(Nodes::Node* node, const std::string& prevTreeName);

        void GetSiblingNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output);

        void GetParentNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output);

        void GetAccessibleNodes(const std::string& sourceNode, std::vector<Nodes::Node*>& output);

        Nodes::Node* GetParent(const std::string& sourceNode);

        std::vector<std::string> GetConstructableNodes();
        std::vector<std::string> GetParametersOfType(std::function<bool(mo::TypeInfo)> selector);
    private:
        NodeFactory();
        virtual ~NodeFactory();
        void printTreeHelper(std::stringstream& tree, int level, Nodes::Node* node);
        void onNodeRecompile(Nodes::Node* node);
        virtual void onConstructorsAdded();

        std::vector<rcc::weak_ptr<Nodes::Node>> nodes;
        std::map<std::string, std::vector<char const*>>        m_nodeInfoMap;
    }; // class NodeManager
}
