#include "Aquila/rcc/SystemTable.hpp"

SystemTable::SystemTable()
{

}
void SystemTable::deleteSingleton(mo::TypeInfo type)
{
    g_singletons.erase(type);
}
void SystemTable::cleanUp()
{
    g_singletons.clear();
}