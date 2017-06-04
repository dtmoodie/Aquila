#pragma once
#include <Aquila/core/detail/Export.hpp>
#include <Aquila/nodes/NodeInfo.hpp>
#include <MetaObject/core/detail/HelperMacros.hpp>
#include <MetaObject/object/MetaObjectInfo.hpp>

#include <vector>
#include <string>

namespace mo
{
    class IVariableManager;
}
namespace aq
{
    namespace Nodes
    {
        struct AQUILA_EXPORTS NodeInfo : virtual public mo::IMetaObjectInfo
        {
            std::string Print(IObjectInfo::Verbosity verbosity = IObjectInfo::INFO) const;
            // Get the organizational hierarchy of this node, ie Image -> Processing -> ConvertToGrey
            virtual std::vector<std::string> getNodeCategory() const = 0;

            // List of nodes that need to be in the direct parental tree of this node, in required order
            virtual std::vector<std::vector<std::string>> getParentalDependencies() const = 0;

            // List of nodes that must exist in this data stream, but do not need to be in the direct parental tree of this node
            virtual std::vector<std::vector<std::string>> getNonParentalDependencies() const = 0;

            // Given the variable manager for a datastream, look for missing dependent variables and return a list of candidate nodes that provide those variables
            virtual std::vector<std::string> checkDependentVariables(mo::IVariableManager* var_manager_) const = 0;
        };
    }
}

template<class T> struct getNodeCategoryHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasNodeCategory, getNodeCategory, std::vector<std::string>(*)(void));
    template<class U>
    static std::vector<std::string> helper(typename std::enable_if<HasNodeCategory<U>::value, void>::type* = 0)
    {
        return U::getNodeCategory();
    }
    template<class U>
    static std::vector<std::string> helper(typename std::enable_if<!HasNodeCategory<U>::value, void>::type* = 0)
    {
        return std::vector<std::string>(1, std::string(U::GetTypeNameStatic()));
    }

    static std::vector<std::string> get()
    {
        return helper<T>();
    }
};

template<class T> struct GetParentDepsHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasParentDeps, getParentalDependencies, std::vector<std::vector<std::string>>(*)(void));
    template<class U>
    static std::vector<std::vector<std::string>> helper(typename std::enable_if<HasParentDeps<U>::value, void>::type* = 0)
    {
        return U::getParentalDependencies();
    }
    template<class U>
    static std::vector<std::vector<std::string>> helper(typename std::enable_if<!HasParentDeps<U>::value, void>::type* = 0)
    {
        return std::vector<std::vector<std::string>>();
    }

    static std::vector<std::vector<std::string>> get()
    {
        return helper<T>();
    }
};

template<class T> struct GetNonParentDepsHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasNonParentDeps, getNonParentalDependencies, std::vector<std::vector<std::string>>(*)(void));
    template<class U>
    static std::vector<std::vector<std::string>> helper(typename std::enable_if<HasNonParentDeps<U>::value, void>::type* = 0)
    {
        return U::getParentalDependencies();
    }
    template<class U>
    static std::vector<std::vector<std::string>> helper(typename std::enable_if<!HasNonParentDeps<U>::value, void>::type* = 0)
    {
        return std::vector<std::vector<std::string>>();
    }

    static std::vector<std::vector<std::string>> get()
    {
        return helper<T>();
    }
};


template<class T> struct GetDepVarHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasDepVar, checkDependentVariables, std::vector<std::string>(*)(mo::IVariableManager*));
    template<class U>
    static std::vector<std::vector<std::string>> helper(mo::IVariableManager* mgr, typename std::enable_if<HasDepVar<U>::value, void>::type* = 0)
    {
        return U::checkDependentVariables(mgr);
    }
    template<class U>
    static std::vector<std::string> helper(mo::IVariableManager* mgr, typename std::enable_if<!HasDepVar<U>::value, void>::type* = 0)
    {
        return std::vector<std::string>();
    }

    static std::vector<std::string> get(mo::IVariableManager* mgr)
    {
        return helper<T>(mgr);
    }
};

namespace mo
{
    // Specialization for FrameGrabber derived classes to pickup extra fields that are needed
    template<class Type>
    struct MetaObjectInfoImpl<Type, aq::Nodes::NodeInfo>: public aq::Nodes::NodeInfo
    {
        std::vector<std::string> getNodeCategory() const
        {
            return getNodeCategoryHelper<Type>::get();
        }

        // List of nodes that need to be in the direct parental tree of this node, in required order
        std::vector<std::vector<std::string>> getParentalDependencies() const
        {
            return GetParentDepsHelper<Type>::get();
        }

        // List of nodes that must exist in this data stream, but do not need to be in the direct parental tree of this node
        std::vector<std::vector<std::string>> getNonParentalDependencies() const
        {
            return GetNonParentDepsHelper<Type>::get();
        }

        // Given the variable manager for a datastream, look for missing dependent variables and return a list of candidate nodes that provide those variables
        std::vector<std::string> checkDependentVariables(mo::IVariableManager* var_manager_) const
        {
            return GetDepVarHelper<Type>::get(var_manager_);
        }
    };
}
