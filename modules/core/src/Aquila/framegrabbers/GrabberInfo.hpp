#ifndef AQUILA_GRABBER_INFO_HPP
#define AQUILA_GRABBER_INFO_HPP

#include "IFrameGrabber.hpp"
namespace grabbers
{
    DEFINE_HAS_STATIC_FUNCTION(HasCanLoad, canLoad, aq::nodes::Priority_t, const std::string&);

    template <class U>
    aq::nodes::Priority_t hasLoadHelper(const std::string& path,
                                        typename std::enable_if<HasCanLoad<U>::value, void>::type* = 0)
    {
        return U::canLoad(path);
    }

    template <class U>
    aq::nodes::Priority_t hasLoadHelper(const std::string& path,
                                        typename std::enable_if<!HasCanLoad<U>::value, void>::type* = 0)
    {
        MO_LOG(trace, "{} Does not have a canLoad function defined", U::GetTypeNameStatic());
        (void)path;
        return 0;
    }

    DEFINE_HAS_STATIC_FUNCTION(HasListPaths, listPaths, void, std::vector<std::string>&);

    template <class U>
    void hasListHelper(std::vector<std::string>& paths,
                       typename std::enable_if<HasListPaths<U>::value, void>::type* = 0)
    {
        U::listPaths(paths);
    }

    template <class U>
    void hasListHelper(std::vector<std::string>& path,
                       typename std::enable_if<!HasListPaths<U>::value, void>::type* = 0)
    {
        (void)path;
    }

    DEFINE_HAS_STATIC_FUNCTION(HasTimeout, loadTimeout, aq::nodes::Timeout_t);

    template <class U>
    aq::nodes::Timeout_t hasTimeoutHelper(typename std::enable_if<HasTimeout<U>::value, void>::type* = 0)
    {
        return U::loadTimeout();
    }

    template <class U>
    aq::nodes::Timeout_t hasTimeoutHelper(typename std::enable_if<!HasTimeout<U>::value, void>::type* = 0)
    {
        return 1000;
    }
} // namespace grabbers

namespace mo
{
    template <class Type>
    struct MetaObjectInfoImpl<Type, aq::nodes::GrabberInfo> : public aq::nodes::GrabberInfo
    {
        enum
        {
            HAS_CANLOAD = grabbers::HasCanLoad<Type>::value,
            HAS_LISTPATHS = grabbers::HasListPaths<Type>::value,
            HAS_TIMEOUT = grabbers::HasTimeout<Type>::value
        };

        aq::nodes::Priority_t canLoad(const std::string& path) const override
        {
            return grabbers::hasLoadHelper<Type>(path);
        }

        void listPaths(std::vector<std::string>& paths) const override
        {
            grabbers::hasListHelper<Type>(paths);
        }

        aq::nodes::Timeout_t timeout() const override
        {
            return grabbers::hasTimeoutHelper<Type>();
        }
    };
} // namespace mo
#endif // AQUILA_GRABBER_INFO_HPP
