#pragma once
namespace mo
{
    template<class T, int N, typename Enable = void> struct MetaObjectPolicy: public MetaObjectPolicy<T, N - 1>
    {
        MetaObjectPolicy() :
            MetaObjectPolicy<T, N - 1>() 
        {
        }
    };
    template<class T> struct MetaObjectPolicy<T, 0, void>
    {
        MetaObjectPolicy()
        {
        }
    };
}
