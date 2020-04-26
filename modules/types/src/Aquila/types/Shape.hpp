#include <Eigen/Core>
#include <ct/types/eigen.hpp>

namespace aq
{
    template <uint8_t D>
    struct Shape : Eigen::Array<uint32_t, D, 1>
    {
        using Super = Eigen::Array<uint32_t, D, 1>;

        Shape()
            : Super()
        {
        }

        template <class... ARGS>
        Shape(ARGS&&... args)
            : Super(std::forward<ARGS>(args)...)
        {
        }

        template <class OtherDerived>
        Shape(const Eigen::ArrayBase<OtherDerived>& other)
            : Super(other)
        {
        }

        template <class OtherDerived>
        Shape<D>& operator=(const Eigen::ArrayBase<OtherDerived>& other)
        {
            this->Super::operator=(other);
            return *this;
        }

        size_t numel() const
        {
            size_t out = 1;
            for (uint8_t i = 0; i < D; ++i)
            {
                out *= (*this)[i];
            }
            return out;
        }
    };
}

namespace ct
{
    // clang-format off
    template <uint8_t D>
    struct ReflectImpl<aq::Shape<D>>
    {
        static constexpr int SPECIALIZED = true;
        using DataType = aq::Shape<D>;
        using BaseTypes = ct::VariadicTypedef<Eigen::Array<uint32_t, D, 1>>;
        static constexpr StringView getName()
        {
            return GetName<DataType>::getName();
        }

        REFLECT_STUB
            // PROPERTY(data, &Reflect<DataType>::getData, &Reflect<DataType>::getDataMutable)
            // PROPERTY_WITH_FLAG(COMPILE_TIME_CONSTANT, shape, &Reflect<DataType>::shape)
            PROPERTY_WITH_FLAG(Flags::COMPILE_TIME_CONSTANT, size)
            PROPERTY_WITH_FLAG(Flags::COMPILE_TIME_CONSTANT, cols)
            PROPERTY_WITH_FLAG(Flags::COMPILE_TIME_CONSTANT, rows)
        REFLECT_INTERNAL_END;

        static constexpr auto end()
        {
            return ct::Indexer<NUM_FIELDS - 1>();
        }
    };
// clang-format on
}
