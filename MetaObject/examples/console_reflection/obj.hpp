#pragma once


#include <MetaObject/MetaObject.hpp>

/*!
 * \brief The ExampleInterfaceInfo struct for static information about objects that inherit from the ExampleInterface class
 */
struct ExampleInterfaceInfo: public mo::IMetaObjectInfo
{
    virtual void PrintHelp() = 0;
};


namespace mo
{
    /*!
     *  This template specialization deals with the concrete object 'Type' which in this example
     *  must have a static function called PrintHelp.
     */
    template<class Type>
    struct MetaObjectInfoImpl<Type, ExampleInterfaceInfo>: public ExampleInterfaceInfo
    {
        /*!
         * \brief PrintHelp calls the static function PrintHelp in the concrete implementation
         *        'Type'
         */
        virtual void PrintHelp()
        {
            return Type::PrintHelp();
        }
    };
}

/*!
 * \brief The ExampleInterface class contains one virtual member foo and the typedef InterfaceInfo
 *
 */
class ExampleInterface: public TInterface<ctcrc32("ExampleInterface"), mo::IMetaObject>
{
public:
    /*!
     * \brief InterfaceInfo typedef allows for the MetaObjectInfo templated class in MetaObject/MetaObjectInfo.hpp
     *        to detect the correct object info interface to inherit from
     */
    typedef ExampleInterfaceInfo InterfaceInfo;

    // These macros are needed to initialize some reflection code

    MO_BEGIN(ExampleInterface)
    MO_END

    // The one virtual function to be called from this interface.
    virtual void foo() = 0;
};
