#pragma once
#include "Aquila/core/detail/Export.hpp"
#include "MetaObject/detail/TypeInfo.hpp"
#include <RuntimeObjectSystem/IObject.h>
#include <RuntimeObjectSystem/shared_ptr.hpp>
#include <functional>
#include <map>
#include <opencv2/core/persistence.hpp>
#include <string>
#include <vector>

namespace aq
{
    namespace nodes
    {
        class INode;
        class Node;
        struct NodeInfo;
    }
    class IGraph;

    class AQUILA_EXPORTS NodeFactory
    {
      public:
        static NodeFactory* Instance();

        void RegisterNodeInfo(const char* nodeName, std::vector<char const*>& nodeInfo);

        const nodes::NodeInfo* getNodeInfo(std::string& nodeName);

        rcc::shared_ptr<nodes::INode> addNode(const std::string& nodeName);

        // Adds a node by name to the data stream or the parent node.
        std::vector<rcc::shared_ptr<nodes::INode>> addNode(const std::string& nodeName, IGraph* parentStream);
        std::vector<rcc::shared_ptr<nodes::INode>> addNode(const std::string& nodeName, nodes::INode* parentNode);

        void PrintNodeTree(std::string* ret = nullptr);
        void SaveTree(const std::string& fileName);
        std::string getNodeFile(const ObjectId& id);

        nodes::INode* getNode(const ObjectId& id);
        nodes::INode* getNode(const std::string& treeName);
        bool removeNode(const std::string& nodeName);
        bool removeNode(ObjectId oid);

        void UpdateTreeName(nodes::Node* node, const std::string& prevTreeName);

        void GetSiblingNodes(const std::string& sourceNode, std::vector<nodes::Node*>& output);

        void GetParentNodes(const std::string& sourceNode, std::vector<nodes::Node*>& output);

        void GetAccessibleNodes(const std::string& sourceNode, std::vector<nodes::Node*>& output);

        nodes::Node* GetParent(const std::string& sourceNode);

        std::vector<std::string> GetConstructableNodes();
        std::vector<std::string> GetParametersOfType(std::function<bool(mo::TypeInfo)> selector);

      private:
        NodeFactory();
        virtual ~NodeFactory();
        void printTreeHelper(std::stringstream& tree, int level, nodes::INode* node);
        void onNodeRecompile(nodes::Node* node);
        virtual void onConstructorsAdded();

        std::vector<rcc::weak_ptr<nodes::INode>> m_nodes;
        std::map<std::string, std::vector<char const*>> m_nodeInfoMap;
    }; // class NodeManager
}
