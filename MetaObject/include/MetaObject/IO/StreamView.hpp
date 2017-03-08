#pragma once
#include <streambuf>

namespace mo
{
    class StreamView : public std::streambuf 
    {
    public:
        StreamView(char *data, size_t size) 
        {
            this->setg(data, data, data + size);
        }
    };
}