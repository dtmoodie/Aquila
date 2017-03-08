#pragma once

namespace Parameters
{
    class IRangedParameter
    {
        public:
        // Rails the value to within the min / max range
        virtual void RailValue() = 0;        
    };
} // namespace Parameters