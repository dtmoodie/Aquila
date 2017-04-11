#pragma once

#include "IFrameGrabber.hpp"
namespace Grabbers
{
    DEFINE_HAS_STATIC_FUNCTION(HasCanLoad, CanLoad, int(*)(const std::string&));
    template<class U> int HasLoadHelper(const std::string& path, typename std::enable_if<HasCanLoad<U>::value, void>::type* = 0)
    {
        return U::CanLoad(path);
    }
    template<class U> int HasLoadHelper(const std::string& path, typename std::enable_if<!HasCanLoad<U>::value, void>::type* = 0)
    {
        return 0;
    }
    DEFINE_HAS_STATIC_FUNCTION(HasListPaths, ListPaths, void(*)(std::vector<std::string>&));
    template<class U> void HasListHelper(std::vector<std::string>& paths, typename std::enable_if<HasListPaths<U>::value, void>::type* = 0)
    {
        U::ListPaths(paths);
    }
    template<class U> void HasListHelper(std::vector<std::string>& path, typename std::enable_if<!HasListPaths<U>::value, void>::type* = 0)
    {
        
    }
    DEFINE_HAS_STATIC_FUNCTION(HasTimeout, Timeout, int(*)(void));
    template<class U> int HasTimeoutHelper(typename std::enable_if<HasListPaths<U>::value, void>::type* = 0)
    {
        return U::Timeout();
    }
    template<class U> int HasTimeoutHelper(typename std::enable_if<!HasListPaths<U>::value, void>::type* = 0)
    {
        return 1000;
    }
}


namespace mo
{
    template<class Type>
    struct MetaObjectInfoImpl<Type, aq::Nodes::GrabberInfo> : public aq::Nodes::GrabberInfo
    {
        int CanLoad(const std::string& path) const
        {
            return Grabbers::HasLoadHelper<Type>(path);
        }
        void ListPaths(std::vector<std::string>& paths) const
        {
            Grabbers::HasListHelper<Type>(paths);
        }
        int Timeout() const
        {
            return Grabbers::HasTimeoutHelper<Type>();
        }
    };
} // namespace mo