#pragma once
#include "MetaObjectMacrosImpl.hpp"


/* 
   These two macros (MO_BEGIN kept for backwards compatibility) are used to define an 
   interface base class.
*/
#define MO_BEGIN(CLASS_NAME) MO_BEGIN_1(CLASS_NAME, __COUNTER__)
#define MO_BASE(CLASS_NAME) MO_BEGIN_1(CLASS_NAME, __COUNTER__)


/*
    These two macros are used for defining a concrete class that has a valid implementation
*/
#define MO_DERIVE(CLASS_NAME, ...) MO_DERIVE_(__COUNTER__, CLASS_NAME, __VA_ARGS__)
#define MO_CONCRETE(CLASS_NAME, ...) MO_DERIVE_(__COUNTER__, CLASS_NAME, __VA_ARGS__)

/*
   This macro is used for defining a abstract class that derives from N interfaces without a
   concrete implementation
*/
#define MO_ABSTRACT(CLASS_NAME, ...) MO_ABSTRACT_(__COUNTER__, CLASS_NAME, __VA_ARGS__)

/*
    This macro is used for marking the end of a class definition block
*/
#define MO_END MO_END_(__COUNTER__)




