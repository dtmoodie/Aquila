#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/map.hpp"
#include "MetaObject/Parameters/IO/TextPolicy.hpp"
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"

INSTANTIATE_META_PARAMETER(std::vector<int>);
INSTANTIATE_META_PARAMETER(std::vector<unsigned short>);
INSTANTIATE_META_PARAMETER(std::vector<unsigned int>);
INSTANTIATE_META_PARAMETER(std::vector<char>);
INSTANTIATE_META_PARAMETER(std::vector<unsigned char>);
INSTANTIATE_META_PARAMETER(std::vector<float>);
INSTANTIATE_META_PARAMETER(std::vector<double>);
INSTANTIATE_META_PARAMETER(std::vector<std::string>);
