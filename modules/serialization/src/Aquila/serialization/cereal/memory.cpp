//#include "MetaObject/serialization/CerealPolicy.hpp"
//#include <Aquila/serialization/cereal/memory.hpp>
#include "MetaObject/params/detail/MetaParamImpl.hpp"
#include "MetaObject/serialization/Policy.hpp"
#include "MetaObject/serialization/SerializationFactory.hpp"
#include "MetaObject/serialization/Serializer.hpp"
#include <Aquila/nodes/Node.hpp>
#include <cereal/types/vector.hpp>

using namespace aq;
using namespace aq::nodes;
#include "MetaObject/params/traits/MemoryTraits.hpp"

INSTANTIATE_META_PARAM(rcc::shared_ptr<Node>);
INSTANTIATE_META_PARAM(rcc::weak_ptr<Node>);
INSTANTIATE_META_PARAM(std::vector<rcc::shared_ptr<Node> >);
INSTANTIATE_META_PARAM(std::vector<rcc::weak_ptr<Node> >);
