#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/Parameters/UI/Qt/OpenCV.hpp"
#include "MetaObject/Parameters/UI/Qt/Containers.hpp"
#include "MetaObject/Parameters/UI/Qt/TParameterProxy.hpp"
#include "MetaObject/Parameters/Buffers/CircularBuffer.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/map.hpp"
#include "MetaObject/Parameters/IO/CerealPolicy.hpp"
#include "MetaObject/Parameters/IO/TextPolicy.hpp"

#ifdef emit
#undef emit
#endif
#ifdef HAVE_WT
#define WT_NO_SLOT_MACROS
#include "MetaObject/Parameters/UI/Wt/POD.hpp"
#include "MetaObject/Parameters/UI/Wt/String.hpp"
#include "MetaObject/Parameters/UI/Wt/IParameterProxy.hpp"

#endif
#include <cereal/types/string.hpp>
#include <cereal/types/map.hpp>
#include "instantiate.hpp"


INSTANTIATE_META_PARAMETER(bool);
INSTANTIATE_META_PARAMETER(int);
INSTANTIATE_META_PARAMETER(unsigned short);
INSTANTIATE_META_PARAMETER(unsigned int);
INSTANTIATE_META_PARAMETER(char);
INSTANTIATE_META_PARAMETER(unsigned char);
INSTANTIATE_META_PARAMETER(float);
INSTANTIATE_META_PARAMETER(double);
INSTANTIATE_META_PARAMETER(std::string);
typedef std::map<std::string, std::string> StringMap;
INSTANTIATE_META_PARAMETER(StringMap);

void mo::instantiations::initialize()
{
    
}
