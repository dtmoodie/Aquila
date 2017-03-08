#pragma once
#include "MetaObject/Detail/Export.hpp"
#include <cstddef>
namespace mo
{
    size_t MO_EXPORTS GetThisThread();
    class MO_EXPORTS ThreadRegistry
    {
    public:
        enum ThreadType
        {
            GUI,
            ANY
        };
        void RegisterThread(ThreadType type, size_t id = GetThisThread());
        size_t GetThread(int type);

        static ThreadRegistry* Instance();
    private:
        ThreadRegistry();
        ~ThreadRegistry();
        
        struct impl;
        impl* _pimpl;
    };
}
