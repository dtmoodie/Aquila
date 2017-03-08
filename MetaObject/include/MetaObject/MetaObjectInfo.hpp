#pragma once
#include "IMetaObjectInfo.hpp"
#include "MetaObject/Detail/HelperMacros.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/MetaObjectInfoDatabase.hpp"
#include <type_traits>



namespace mo
{
	// Static object information available for each meta object
	// Used for static introspection

    // Specialize this for each class which requires additional fields
    template<class Type, class InterfaceInfo>
    struct MetaObjectInfoImpl: public InterfaceInfo
    {
    };

    template<class T> 
    struct MetaObjectInfo: public MetaObjectInfoImpl<T, typename T::InterfaceInfo>
    {
        MetaObjectInfo()
        {
            MetaObjectInfoDatabase::Instance()->RegisterInfo(this);   
        }
        static void GetParameterInfoStatic(std::vector<ParameterInfo*>& info)
        {
            T::GetParameterInfoStatic(info);
        }
        static void GetSignalInfoStatic(std::vector<SignalInfo*>& info)
        {
            T::GetSignalInfoStatic(info);
        }
        static void GetSlotInfoStatic(std::vector<SlotInfo*>& info)
        {
            T::GetSlotInfoStatic(info);
        }
		static std::string                 TooltipStatic()
        {
            return _get_tooltip_helper<T>();
        }
        static std::string                 DescriptionStatic()
        {
            return _get_description_helper<T>();
        }
        static TypeInfo                    GetTypeInfoStatic()
        {
            return TypeInfo(typeid(typename T::BASE_CLASS));
        }
        std::vector<ParameterInfo*>        GetParameterInfo() const
        {
            std::vector<ParameterInfo*> info;
            GetParameterInfoStatic(info);
            return info;
        }
		std::vector<SignalInfo*>           GetSignalInfo() const
        {
            std::vector<SignalInfo*> info;
            GetSignalInfoStatic(info);
            return info;
        }
		std::vector<SlotInfo*>             GetSlotInfo() const
        {
            std::vector<SlotInfo*> info;
            GetSlotInfoStatic(info);
            return info;
        }
		std::string                        GetObjectTooltip() const
        {
            return TooltipStatic();
        }
		std::string                        GetObjectHelp() const
        {
            return DescriptionStatic();
        }
        TypeInfo                           GetTypeInfo() const
        {
            return GetTypeInfoStatic();
        }
        std::string                        GetObjectName() const
        {
            return T::GetTypeNameStatic();
        }
        unsigned int                       GetInterfaceId() const
        {
            return T::s_interfaceID;
        }
    private:
        DEFINE_HAS_STATIC_FUNCTION(HasTooltip, V::GetTooltipStatic, std::string(*)(void));
        DEFINE_HAS_STATIC_FUNCTION(HasDescription, V::GetDescriptionStatic, std::string(*)(void));
        template<class U> static std::string _get_tooltip_helper(typename std::enable_if<HasTooltip<U>::value, void>::type* = 0)
        {
            return U::GetTooltipStatic();
        }
        template<class U> static std::string _get_tooltip_helper(typename std::enable_if<!HasTooltip<U>::value, void>::type* = 0)
        {
            return "";
        }
        template<class U> static std::string _get_description_helper(typename std::enable_if<HasDescription<U>::value, void>::type* = 0)
        {
            return U::GetDescriptionStatic();
        }
        template<class U> static std::string _get_description_helper(typename std::enable_if<!HasDescription<U>::value, void>::type* = 0)
        {
            return "";
        }
    };
}

