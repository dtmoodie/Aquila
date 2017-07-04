#pragma once

#include "IFrameGrabber.hpp"
namespace Grabbers
{
    DEFINE_HAS_STATIC_FUNCTION(HasCanLoad, canLoad, int(*)(const std::string&));
    template<class U> int HasLoadHelper(const std::string& path, typename std::enable_if<HasCanLoad<U>::value, void>::type* = 0)
    {
        return U::canLoad(path);
    }
    template<class U> int HasLoadHelper(const std::string& path, typename std::enable_if<!HasCanLoad<U>::value, void>::type* = 0)
    {
        (void)path;
        return 0;
    }
    DEFINE_HAS_STATIC_FUNCTION(HasListPaths, listPaths, void(*)(std::vector<std::string>&));
    template<class U> void HasListHelper(std::vector<std::string>& paths, typename std::enable_if<HasListPaths<U>::value, void>::type* = 0)
    {
        U::listPaths(paths);
    }
    template<class U> void HasListHelper(std::vector<std::string>& path, typename std::enable_if<!HasListPaths<U>::value, void>::type* = 0)
    {
        (void)path;
    }
    DEFINE_HAS_STATIC_FUNCTION(HasTimeout, loadTimeout, int(*)(void));
    template<class U> int HasTimeoutHelper(typename std::enable_if<HasTimeout<U>::value, void>::type* = 0)
    {
        return U::loadTimeout();
    }
    template<class U> int HasTimeoutHelper(typename std::enable_if<!HasTimeout<U>::value, void>::type* = 0)
    {
        return 1000;
    }
}


namespace mo
{
    template<class Type>
    struct MetaObjectInfoImpl<Type, aq::nodes::GrabberInfo> : public aq::nodes::GrabberInfo
    {
        enum{
            HAS_CANLOAD = Grabbers::HasCanLoad<Type>::value,
            HAS_LISTPATHS = Grabbers::HasListPaths<Type>::value,
            HAS_TIMEOUT = Grabbers::HasTimeout<Type>::value
        };
        int canLoad(const std::string& path) const
        {
            return Grabbers::HasLoadHelper<Type>(path);
        }
        void listPaths(std::vector<std::string>& paths) const
        {
            Grabbers::HasListHelper<Type>(paths);
        }
        int timeout() const
        {
            return Grabbers::HasTimeoutHelper<Type>();
        }
    };
} // namespace mo
