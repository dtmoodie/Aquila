#include "Aquila/rendering/RenderingEngine.h"
#include "RuntimeObjectSystem/ObjectInterfacePerModule.h"
#include "Aquila/rcc/SystemTable.hpp"

using namespace aq;


void IRenderObjectFactory::RegisterConstructorStatic(std::shared_ptr<IRenderObjectConstructor> constructor)
{
    auto systemTable = PerModuleInterface::GetInstance()->GetSystemTable();
    if (systemTable)
    {
        auto factoryInstance = systemTable->GetSingleton<IRenderObjectFactory>();
        if (factoryInstance)
        {
            factoryInstance->RegisterConstructor(constructor);
        }
    }
}