#ifndef AQUILA_FRAME_GRABBER_INFO_HPP
#define AQUILA_FRAME_GRABBER_INFO_HPP
#include "Aquila/nodes/NodeInfo.hpp"
#include "IFrameGrabber.hpp"

namespace aq
{
    namespace nodes
    {
        class FrameGrabberInfo;
    }
} // namespace aq

// I tried placing these as functions inside of the MetaObjectInfoImpl specialization, but msvc doesn't like that. :(
DEFINE_HAS_STATIC_FUNCTION(HasLoadablePaths, listLoadablePaths, std::vector<std::string>);

template <class T>
struct GetLoadablePathsHelper
{

    template <class U>
    static std::vector<std::string> helper(ct::EnableIf<HasLoadablePaths<U>::value, int32_t> = 0)
    {
        return U::listLoadablePaths();
    }

    template <class U>
    static std::vector<std::string> helper(ct::DisableIf<HasLoadablePaths<U>::value, int32_t> = 0)
    {
        return std::vector<std::string>();
    }

    static std::vector<std::string> get()
    {
        return helper<T>();
    }
};

DEFINE_HAS_STATIC_FUNCTION(HasTimeout, loadTimeout, aq::nodes::Timeout_t);

template <class T>
struct GetTimeoutHelper
{

    enum
    {
        value = HasTimeout<T>::value
    };

    template <class U>
    static aq::nodes::Timeout_t helper(ct::EnableIf<HasTimeout<U>::value, int32_t> = 0)
    {
        return U::loadTimeout();
    }

    template <class U>
    static aq::nodes::Timeout_t helper(ct::DisableIf<HasTimeout<U>::value, int32_t> = 0)
    {
        return aq::nodes::Timeout_t{1000};
    }

    static aq::nodes::Timeout_t get()
    {
        return helper<T>();
    }
};

DEFINE_HAS_STATIC_FUNCTION(HasCanLoad, canLoadPath, aq::nodes::Priority_t, const std::string&);

template <class T>
struct GetCanLoadHelper
{
    enum
    {
        value = HasCanLoad<T>::value
    };

    template <class U>
    static aq::nodes::Priority_t helper(const std::string& doc, ct::EnableIf<HasCanLoad<U>::value, int32_t> = 0)
    {
        return U::canLoadPath(doc);
    }

    template <class U>
    static aq::nodes::Priority_t helper(const std::string&, ct::DisableIf<HasCanLoad<U>::value, int32_t> = 0)
    {
        return aq::nodes::Priority_t{0};
    }

    static aq::nodes::Priority_t get(const std::string& doc)
    {
        return helper<T>(doc);
    }
};

namespace mo
{
    // Specialization for FrameGrabber derived classes to pickup extra fields that are needed
    template <class Type>
    struct MetaObjectInfoImpl<Type, aq::nodes::FrameGrabberInfo> : public aq::nodes::FrameGrabberInfo
    {
        aq::nodes::Timeout_t loadTimeout() const
        {
            return GetTimeoutHelper<Type>::get();
        }

        std::vector<std::string> listLoadablePaths() const
        {
            return GetLoadablePathsHelper<Type>::get();
        }

        aq::nodes::Priority_t canLoadPath(const std::string& document) const
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
} // namespace mo
#endif // AQUILA_FRAME_GRABBER_INFO_HPP
