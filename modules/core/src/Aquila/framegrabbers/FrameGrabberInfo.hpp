#pragma once
#include "Aquila/nodes/NodeInfo.hpp"
#include "IFrameGrabber.hpp"
namespace aq
{
    namespace nodes
    {
        class FrameGrabberInfo;
    }
}

// I tried placing these as functions inside of the MetaObjectInfoImpl specialization, but msvc doesn't like that. :(
template <class T>
struct GetLoadablePathsHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasLoadablePaths, listLoadablePaths, std::vector<std::string> (*)(void));
    template <class U>
    static std::vector<std::string> helper(typename std::enable_if<HasLoadablePaths<U>::value, void>::type* = 0)
    {
        return U::listLoadablePaths();
    }
    template <class U>
    static std::vector<std::string> helper(typename std::enable_if<!HasLoadablePaths<U>::value, void>::type* = 0)
    {
        return std::vector<std::string>();
    }

    static std::vector<std::string> get()
    {
        return helper<T>();
    }
};

template <class T>
struct GetTimeoutHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasTimeout, loadTimeout, int (*)(void));
    template <class U>
    static int helper(typename std::enable_if<HasTimeout<U>::value, void>::type* = 0)
    {
        return U::loadTimeout();
    }
    template <class U>
    static int helper(typename std::enable_if<!HasTimeout<U>::value, void>::type* = 0)
    {
        return 1000;
    }

    static int get()
    {
        return helper<T>();
    }
};

template <class T>
struct GetCanLoadHelper
{
    DEFINE_HAS_STATIC_FUNCTION(HasCanLoad, canLoadPath, int (*)(const std::string&));
    template <class U>
    static int helper(const std::string& doc, typename std::enable_if<HasCanLoad<U>::value, void>::type* = 0)
    {
        return U::canLoadPath(doc);
    }
    template <class U>
    static int helper(const std::string&, typename std::enable_if<!HasCanLoad<U>::value, void>::type* = 0)
    {
        return 0;
    }

    static int get(const std::string& doc)
    {
        return helper<T>(doc);
    }
    enum
    {
        value = HasCanLoad<T>::value
    };
};

namespace mo
{
    // Specialization for FrameGrabber derived classes to pickup extra fields that are needed
    template <class Type>
    struct MetaObjectInfoImpl<Type, aq::nodes::FrameGrabberInfo> : public aq::nodes::FrameGrabberInfo
    {
        int loadTimeout() const
        {
            return GetTimeoutHelper<Type>::get();
        }

        std::vector<std::string> listLoadablePaths() const
        {
            return GetLoadablePathsHelper<Type>::get();
        }

        int canLoadPath(const std::string& document) const
        {
            if (GetCanLoadHelper<Type>::value)
                return GetCanLoadHelper<Type>::get(document);
            else
                return aq::nodes::FrameGrabberInfo::canLoadPath(document);
        }

        std::vector<std::string> getNodeCategory() const
        {
            return getNodeCategoryHelper<Type>::get();
        }

        // List of nodes that need to be in the direct parental tree of this node, in required order
        std::vector<std::vector<std::string>> getParentalDependencies() const
        {
            return GetParentDepsHelper<Type>::get();
        }

        // List of nodes that must exist in this data stream, but do not need to be in the direct parental tree of this
        // node
        std::vector<std::vector<std::string>> getNonParentalDependencies() const
        {
            return GetNonParentDepsHelper<Type>::get();
        }

        // Given the variable manager for a Graph, look for missing dependent variables and return a list of candidate
        // nodes that provide those variables
        std::vector<std::string> checkDependentVariables(mo::IParamServer* var_manager_) const
        {
            return GetDepVarHelper<Type>::get(var_manager_);
        }
    };
}
