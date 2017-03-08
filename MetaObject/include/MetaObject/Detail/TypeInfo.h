// The Loki Library
// Copyright (c) 2001 by Andrei Alexandrescu
// This code accompanies the book:
// Alexandrescu, Andrei. "Modern C++ Design: Generic Programming and Design
//     Patterns Applied". Copyright (c) 2001. Addison-Wesley.
// Permission to use, copy, modify, distribute and sell this software for any
//     purpose is hereby granted without fee, provided that the above copyright
//     notice appear in all copies and that both that copyright notice and this
//     permission notice appear in supporting documentation.
// The author or Addison-Wesley Longman make no representations about the
//     suitability of this software for any purpose. It is provided "as is"
//     without express or implied warranty.
#ifndef LOKI_LOKITYPEINFO_INC_
#define LOKI_LOKITYPEINFO_INC_

// $Id: LokiTypeInfo.h 748 2006-10-17 19:49:08Z syntheticpp $


#include <typeinfo>
#include <cassert>
#include <string>
// Needed to demangle GCC's name mangling.
#ifndef _MSC_VER
#include <cstdlib>
#include <memory>
#include <cxxabi.h>
#endif

namespace mo
{
// class TypeInfo
// Purpose: offer a first-class, comparable wrapper over std::type_info

    class TypeInfo
    {
     public:
        // Constructors
      TypeInfo(); // needed for containers
         TypeInfo(const std::type_info&); // non-explicit

         // Access for the wrapped std::type_info
         const std::type_info& Get() const;
         // Compatibility functions
        bool before(const TypeInfo& rhs) const;
         std::string name() const;

     private:
         const std::type_info* pInfo_;
     };

    // Implementation

    inline TypeInfo::TypeInfo()
    {
         class Nil {};
         pInfo_ = &typeid(Nil);
        assert(pInfo_);
     }

     inline TypeInfo::TypeInfo(const std::type_info& ti)
     : pInfo_(&ti)
     { assert(pInfo_); }

     inline bool TypeInfo::before(const TypeInfo& rhs) const
    {
         assert(pInfo_);
         // type_info::before return type is int in some VC libraries
         return pInfo_->before(*rhs.pInfo_) != 0;
     }

     inline const std::type_info& TypeInfo::Get() const
     {
         assert(pInfo_);
         return *pInfo_;
     }

     inline std::string TypeInfo::name() const
     {
         assert(pInfo_);
#ifdef _MSC_VER
         return pInfo_->name();
#else
         int status = -4; // some arbitrary value to eliminate the compiler warning

         // enable c++11 by passing the flag -std=c++11 to g++
         std::unique_ptr<char, void(*)(void*)> res {
             abi::__cxa_demangle(pInfo_->name(), NULL, NULL, &status),
             std::free
         };

         return (status==0) ? res.get() : pInfo_->name() ;
#endif
     }

    // Comparison operators

     inline bool operator==(const TypeInfo& lhs, const TypeInfo& rhs)
     // type_info::operator== return type is int in some VC libraries
     { return (lhs.Get() == rhs.Get()) != 0; }

     inline bool operator<(const TypeInfo& lhs, const TypeInfo& rhs)
     { return lhs.before(rhs); }

     inline bool operator!=(const TypeInfo& lhs, const TypeInfo& rhs)
     { return !(lhs == rhs); }

     inline bool operator>(const TypeInfo& lhs, const TypeInfo& rhs)
     { return rhs < lhs; }

     inline bool operator<=(const TypeInfo& lhs, const TypeInfo& rhs)
     { return !(lhs > rhs); }

     inline bool operator>=(const TypeInfo& lhs, const TypeInfo& rhs)
     { return !(lhs < rhs); }
 }

 #endif // end file guardian
