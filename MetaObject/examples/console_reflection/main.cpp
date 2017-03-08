#include <obj.hpp>

int main()
{
    /*!
     * \brief factory is a pointer to the global object factory
     */
    auto factory = mo::MetaObjectFactory::Instance();

    // call the inlined register translation unit function to register ConcreteImplementation
    // to the global object registry
    factory->RegisterTranslationUnit();

    // Get a list of objects that inherit from ExampleInterface
    auto constructors = factory->GetConstructors(ExampleInterface::s_interfaceID);

    // Print static object info
    for(IObjectConstructor* constructor : constructors)
    {
        IObjectInfo* info = constructor->GetObjectInfo();
        if(ExampleInterfaceInfo* interface_info = dynamic_cast<ExampleInterfaceInfo*>(info))
        {
            interface_info->PrintHelp();
        }
        // Print reflection info
        std::cout <<  info->Print() << std::endl;
    }

    // Construct an object
    mo::IMetaObject* obj = factory->Create("ConcreteImplementation");
    if(ExampleInterface* interface_object = dynamic_cast<ExampleInterface*>(obj))
    {
        interface_object->foo();
    }
    delete obj;
    return 0;
}
