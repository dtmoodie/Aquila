#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include <Aquila/IO/memory.hpp>

#include "MetaObject/IO/Serializer.hpp"
#include "MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp"

#include "MetaObject/IO/Policy.hpp"
#include <cereal/types/vector.hpp>
//#include "MetaObject/Parameters/IO/CerealPolicy.hpp"

using namespace aq;
using namespace aq::Nodes;
/*INSTANTIATE_META_PARAMETER(rcc::shared_ptr<Node>);
INSTANTIATE_META_PARAMETER(rcc::weak_ptr<Node>);
INSTANTIATE_META_PARAMETER(std::vector<rcc::shared_ptr<Node>>);
INSTANTIATE_META_PARAMETER(std::vector<rcc::weak_ptr<Node>>);
*/