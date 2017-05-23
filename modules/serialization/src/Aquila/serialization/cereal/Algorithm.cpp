#include <Aquila/core/Algorithm.hpp>
#include <Aquila/serialization/cereal/JsonArchive.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>
#include <MetaObject/params/detail/MetaParamImpl.hpp>

INSTANTIATE_META_PARAM(std::vector<rcc::shared_ptr<aq::Algorithm>>);
