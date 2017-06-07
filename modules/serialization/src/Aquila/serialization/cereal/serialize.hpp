#pragma once

#include "Aquila/core/detail/Export.hpp"

namespace cereal
{
    class BinaryOutputArchive;
    class BinaryInputArchive;
    class XMLOutputArchive;
    class XMLInputArchive;
    class JSONOutputArchive;
    class JSONInputArchive;
}

namespace aq
{
    namespace Nodes
    {
        class Node;
    }
    AQUILA_EXPORTS bool Serialize(cereal::BinaryOutputArchive& ar, const aq::Nodes::Node* obj);
    AQUILA_EXPORTS bool DeSerialize(cereal::BinaryInputArchive& ar, aq::Nodes::Node* obj);
    AQUILA_EXPORTS bool Serialize(cereal::XMLOutputArchive& ar, const aq::Nodes::Node* obj);
    AQUILA_EXPORTS bool DeSerialize(cereal::XMLInputArchive& ar, aq::Nodes::Node* obj);
    AQUILA_EXPORTS bool Serialize(cereal::JSONOutputArchive& ar, const aq::Nodes::Node* obj);
    AQUILA_EXPORTS bool DeSerialize(cereal::JSONInputArchive& ar, aq::Nodes::Node* obj);
}