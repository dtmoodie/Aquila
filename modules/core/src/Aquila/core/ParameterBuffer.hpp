#pragma once
#include "IParameterBuffer.hpp"

#include <boost/circular_buffer.hpp>
#include <mutex>
#include <map>

namespace aq{
    class ParameterBuffer : public IParameterBuffer{
        template<typename T> struct FN : public T{
            template<class...U> FN(int frame_number_ = 0) : frame_number(frame_number_) {}
            template<class...U> FN(int frame_number_, const U&... args) : frame_number(frame_number_), T(args...) {}
            FN& operator=(const T& other){
                T::operator=(other);
                return *this;
            }
            template<class A> void serialize(A& ar){
                ar(frame_number);
                ar(*static_cast<T*>(this));
            }

            int frame_number;
        };
        std::map<mo::TypeInfo, std::map<std::string, boost::circular_buffer<FN<boost::any>>>> _parameter_map;
        std::mutex mtx;
        int _initial_size;
    public:
        ParameterBuffer(int size);
        void setBufferSize(int size);
        virtual boost::any& getParameter(mo::TypeInfo type, const std::string& name, int frameNumber);
    };

}
