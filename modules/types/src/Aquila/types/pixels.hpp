#ifndef AQUILA_TYPES_PIXELS_HPP
#define AQUILA_TYPES_PIXELS_HPP
#include <Eigen/Core>
#include <ct/enum.hpp>
#include <ct/reflect_macros.hpp>
#include <type_traits>
#include <utility>
namespace aq
{
    ENUM_BEGIN(PixelFormat, uint8_t)
        ENUM_VALUE(kUNCHANGED, 0)
        ENUM_VALUE(kGRAY, 1)
        ENUM_VALUE(kRGB, kGRAY + 1)
        ENUM_VALUE(kBGR, kRGB + 1)
        ENUM_VALUE(kHSV, kRGB + 1)
        ENUM_VALUE(kHSL, kHSV + 1)
        ENUM_VALUE(kLUV, kHSL + 1)
        ENUM_VALUE(kARGB, kLUV + 1)
        ENUM_VALUE(kRGBA, kARGB + 1)
        ENUM_VALUE(kBGRA, kRGBA + 1)
        ENUM_VALUE(kABGR, kBGRA + 1)
        ENUM_VALUE(kBAYER_BGGR, kABGR + 1)
        ENUM_VALUE(kBAYER_RGGB, kBAYER_BGGR + 1)

        constexpr uint8_t numChannels() const;
    ENUM_END;

    template <class T>
    struct GRAY;
    template <class T>
    struct RGB;
    template <class T>
    struct BGR;
    template <class T>
    struct HSV;
    template <class T>
    struct HSL;
    template <class T>
    struct LUV;
    template <class T>
    struct ARGB;
    template <class T>
    struct RGBA;
    template <class T>
    struct BGRA;
    template <class T>
    struct ABGR;

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    struct PackedPixelBase
    {
        using Scalar_t = SCALAR;
        static constexpr const uint8_t num_channels = CHANNELS;

        template <class... ARGS>
        PackedPixelBase(ARGS&&... args)
            : m_data{std::forward<ARGS>(args)...}
        {
        }

        GRAY<Scalar_t> gray() const;
        RGB<Scalar_t> rgb() const;
        BGR<Scalar_t> bgr() const;
        HSV<Scalar_t> hsv() const;
        HSL<Scalar_t> hsl() const;
        LUV<Scalar_t> luv() const;
        ARGB<Scalar_t> argb() const;
        RGBA<Scalar_t> rgba() const;
        BGRA<Scalar_t> bgra() const;
        ABGR<Scalar_t> abgr() const;

        // Returns h,s,v in terms of bgr values implemented in derived
        // Override by derived that can just return these values
        Scalar_t h() const;
        Scalar_t s() const;
        Scalar_t v() const;

        Scalar_t& operator()(size_t i)
        {
            return m_data[i];
        }

        const Scalar_t& operator()(size_t i) const
        {
            return m_data[i];
        }

        Scalar_t& operator[](size_t i)
        {
            return m_data[i];
        }

        const Scalar_t& operator[](size_t i) const
        {
            return m_data[i];
        }

      private:
        DERIVED* derived();
        const DERIVED* derived() const;
        SCALAR m_data[CHANNELS];
    };

    template <class T>
    struct GRAY : PackedPixelBase<GRAY<T>, T, 1>
    {
        using Super_t = PackedPixelBase<GRAY<T>, T, 1>;
        static constexpr const auto pixel_format = PixelFormat::kGRAY;
        GRAY(T v = 0)
            : PackedPixelBase<GRAY<T>, T, 1>(v)
        {
        }

        const T& r() const
        {
            return (*this)(0);
        }
        const T& g() const
        {
            return (*this)(0);
        }
        const T& b() const
        {
            return (*this)(0);
        }
        T a() const
        {
            return std::is_floating_point<T>::value ? 1 : 255;
        }

        T& r()
        {
            return (*this)(0);
        }
        T& g()
        {
            return (*this)(0);
        }
        T& b()
        {
            return (*this)(0);
        }
    };

    template <class T>
    struct RGB : PackedPixelBase<RGB<T>, T, 3>
    {
        using Super_t = PackedPixelBase<RGB<T>, T, 3>;
        static constexpr const auto pixel_format = PixelFormat::kRGB;
        RGB(const RGB&) = default;
        RGB(T r = 0, T g = 0, T b = 0)
            : Super_t(r, g, b)
        {
        }

        const T& r() const
        {
            return (*this)(0);
        }
        const T& g() const
        {
            return (*this)(1);
        }
        const T& b() const
        {
            return (*this)(2);
        }
        T a() const
        {
            return std::is_floating_point<T>::value ? 1 : 255;
        }

        T& r()
        {
            return (*this)(0);
        }
        T& g()
        {
            return (*this)(1);
        }
        T& b()
        {
            return (*this)(2);
        }
    };

    template <class T>
    struct ARGB : PackedPixelBase<ARGB<T>, T, 4>
    {
        using Super_t = PackedPixelBase<ARGB<T>, T, 4>;
        static constexpr const auto pixel_format = PixelFormat::kARGB;
        ARGB(T a = 0, T r = 0, T g = 0, T b = 0)
            : Super_t(a, r, g, b)
        {
        }
        const T& a() const
        {
            return (*this)(0);
        }
        const T& r() const
        {
            return (*this)(1);
        }
        const T& g() const
        {
            return (*this)(2);
        }
        const T& b() const
        {
            return (*this)(3);
        }

        T& a()
        {
            return (*this)(0);
        }
        T& r()
        {
            return (*this)(1);
        }
        T& g()
        {
            return (*this)(2);
        }
        T& b()
        {
            return (*this)(3);
        }
    };

    template <class T>
    struct RGBA : PackedPixelBase<RGBA<T>, T, 4>
    {
        using Super_t = PackedPixelBase<RGBA<T>, T, 4>;
        static constexpr const auto pixel_format = PixelFormat::kRGBA;
        RGBA(T r = 0, T g = 0, T b = 0, T a = 0)
            : Super_t(r, g, b, a)
        {
        }

        const T& r() const
        {
            return (*this)(0);
        }

        const T& g() const
        {
            return (*this)(1);
        }

        const T& b() const
        {
            return (*this)(2);
        }

        const T& a() const
        {
            return (*this)(3);
        }

        T& r()
        {
            return (*this)(0);
        }

        T& g()
        {
            return (*this)(1);
        }

        T& b()
        {
            return (*this)(2);
        }

        T& a()
        {
            return (*this)(3);
        }
    };

    template <class T>
    struct BGR : PackedPixelBase<BGR<T>, T, 3>
    {
        using Super_t = PackedPixelBase<BGR<T>, T, 3>;
        static constexpr const auto pixel_format = PixelFormat::kBGR;
        BGR(T b = 0, T g = 0, T r = 0)
            : Super_t(b, g, r)
        {
        }

        const T& b() const
        {
            return (*this)(0);
        }

        const T& g() const
        {
            return (*this)(1);
        }

        const T& r() const
        {
            return (*this)(2);
        }

        T a() const
        {
            return std::is_floating_point<T>::value ? 1 : 255;
        }

        T& b()
        {
            return (*this)(0);
        }

        T& g()
        {
            return (*this)(1);
        }

        T& r()
        {
            return (*this)(2);
        }
    };

    template <class T>
    struct ABGR : PackedPixelBase<ABGR<T>, T, 4>
    {
        using Super_t = PackedPixelBase<ABGR<T>, T, 4>;
        static constexpr const auto pixel_format = PixelFormat::kABGR;
        ABGR(T a = 0, T b = 0, T g = 0, T r = 0)
            : Super_t(a, b, g, r)
        {
        }

        const T& a() const
        {
            return (*this)(0);
        }

        const T& b() const
        {
            return (*this)(1);
        }

        const T& g() const
        {
            return (*this)(2);
        }

        const T& r() const
        {
            return (*this)(3);
        }

        T& a()
        {
            return (*this)(0);
        }

        T& b()
        {
            return (*this)(1);
        }

        T& g()
        {
            return (*this)(2);
        }

        T& r()
        {
            return (*this)(3);
        }
    };

    template <class T>
    struct BGRA : PackedPixelBase<BGRA<T>, T, 4>
    {
        using Super_t = PackedPixelBase<BGRA<T>, T, 4>;
        static constexpr const auto pixel_format = PixelFormat::kBGRA;
        BGRA(T b = 0, T g = 0, T r = 0, T a = 0)
            : Super_t(b, g, r, a)
        {
        }

        const T& b() const
        {
            return (*this)(0);
        }

        const T& g() const
        {
            return (*this)(1);
        }

        const T& r() const
        {
            return (*this)(2);
        }

        const T& a() const
        {
            return (*this)(3);
        }

        T& b()
        {
            return (*this)(0);
        }

        T& g()
        {
            return (*this)(1);
        }

        T& r()
        {
            return (*this)(2);
        }

        T& a()
        {
            return (*this)(3);
        }
    };

    template <class T>
    struct HSV : PackedPixelBase<HSV<T>, T, 3>
    {
        using Super_t = PackedPixelBase<HSV<T>, T, 3>;
        static constexpr const auto pixel_format = PixelFormat::kHSV;
        HSV(T h = 0, T s = 0, T v = 0)
            : Super_t(h, s, v)
        {
        }

        const T& h() const
        {
            return (*this)(0);
        }

        const T& s() const
        {
            return (*this)(1);
        }

        const T& v() const
        {
            return (*this)(2);
        }

        T a() const
        {
            return std::is_floating_point<T>::value ? 1 : 255;
        }

        T& h()
        {
            return (*this)(0);
        }

        T& s()
        {
            return (*this)(1);
        }

        T& v()
        {
            return (*this)(2);
        }

        // TODO Calculate r, g, b values from hsv
        T r() const;

        T g() const;

        T b() const;
    };

    template <class T>
    struct HSL : PackedPixelBase<HSL<T>, T, 3>
    {
        using Super_t = PackedPixelBase<HSL<T>, T, 3>;
        static constexpr const auto pixel_format = PixelFormat::kHSL;
        HSL(T h = 0, T s = 0, T l = 0)
            : Super_t(h, s, l)
        {
        }

        const T& h() const
        {
            return (*this)(0);
        }

        const T& s() const
        {
            return (*this)(1);
        }

        const T& l() const
        {
            return (*this)(2);
        }

        T a() const
        {
            return std::is_floating_point<T>::value ? 1 : 255;
        }

        T& h()
        {
            return (*this)(0);
        }

        T& s()
        {
            return (*this)(1);
        }

        T& l()
        {
            return (*this)(2);
        }

        // TODO Calculate r, g, b values from hsl
        T r();
        T g();
        T b();
    };

    template <class T>
    struct LUV : PackedPixelBase<LUV<T>, T, 3>
    {
        using Super_t = PackedPixelBase<LUV<T>, T, 3>;
        static constexpr const auto pixel_format = PixelFormat::kLUV;
        LUV(T h = 0, T s = 0, T l = 0)
            : Super_t(h, s, l)
        {
        }

        const T& l() const
        {
            return (*this)(0);
        }

        const T& u() const
        {
            return (*this)(1);
        }

        const T& v() const
        {
            return (*this)(2);
        }

        T a() const
        {
            return std::is_floating_point<T>::value ? 1 : 255;
        }

        T& l()
        {
            return (*this)(0);
        }

        T& u()
        {
            return (*this)(1);
        }

        T& v()
        {
            return (*this)(2);
        }

        // TODO Calculate r, g, b values from luv
        T r();
        T g();
        T b();
    };

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    GRAY<SCALAR> PackedPixelBase<DERIVED, SCALAR, CHANNELS>::gray() const
    {
        return {0.2989 * derived()->r() + 0.5870 * derived()->g() + 0.1140 * derived()->b()};
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    RGB<SCALAR> PackedPixelBase<DERIVED, SCALAR, CHANNELS>::rgb() const
    {
        return {derived()->r(), derived()->g(), derived()->b()};
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    BGR<SCALAR> PackedPixelBase<DERIVED, SCALAR, CHANNELS>::bgr() const
    {
        return {derived()->b(), derived()->g(), derived()->r()};
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    HSV<SCALAR> PackedPixelBase<DERIVED, SCALAR, CHANNELS>::hsv() const
    {
        return {derived()->h(), derived()->s(), derived()->v()};
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    HSL<SCALAR> PackedPixelBase<DERIVED, SCALAR, CHANNELS>::hsl() const
    {
        return {derived()->h(), derived()->s(), derived()->l()};
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    LUV<SCALAR> PackedPixelBase<DERIVED, SCALAR, CHANNELS>::luv() const
    {
        // TODO
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    ARGB<SCALAR> PackedPixelBase<DERIVED, SCALAR, CHANNELS>::argb() const
    {
        return {derived()->a(), derived()->r(), derived()->g(), derived()->b()};
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    RGBA<SCALAR> PackedPixelBase<DERIVED, SCALAR, CHANNELS>::rgba() const
    {
        return {derived()->r(), derived()->g(), derived()->b(), derived()->a()};
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    BGRA<SCALAR> PackedPixelBase<DERIVED, SCALAR, CHANNELS>::bgra() const
    {
        return {derived()->b(), derived()->g(), derived()->r(), derived()->a()};
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    ABGR<SCALAR> PackedPixelBase<DERIVED, SCALAR, CHANNELS>::abgr() const
    {
        return {derived()->a(), derived()->b(), derived()->g(), derived()->r()};
    }

    // Returns h,s,v in terms of bgr values implemented in derived
    // Override by derived that can just return these values
    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    SCALAR PackedPixelBase<DERIVED, SCALAR, CHANNELS>::h() const
    {
        // TODO
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    SCALAR PackedPixelBase<DERIVED, SCALAR, CHANNELS>::s() const
    {
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    SCALAR PackedPixelBase<DERIVED, SCALAR, CHANNELS>::v() const
    {
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    DERIVED* PackedPixelBase<DERIVED, SCALAR, CHANNELS>::derived()
    {
        return static_cast<DERIVED*>(this);
    }

    template <class DERIVED, class SCALAR, uint8_t CHANNELS>
    const DERIVED* PackedPixelBase<DERIVED, SCALAR, CHANNELS>::derived() const
    {
        return static_cast<const DERIVED*>(this);
    }

    template <class T>
    using AvailablePixelFormats_t =
        ct::VariadicTypedef<GRAY<T>, RGB<T>, BGR<T>, HSV<T>, HSL<T>, LUV<T>, ARGB<T>, RGBA<T>, BGRA<T>, ABGR<T>>;

    template <class T>
    constexpr uint8_t numChannelsHelper(PixelFormat fmt, ct::VariadicTypedef<T>)
    {
        return fmt == T::pixel_format ? T::num_channels : 0;
    }

    template <class T, class... ARGS>
    constexpr uint8_t numChannelsHelper(PixelFormat fmt, ct::VariadicTypedef<T, ARGS...>)
    {
        return fmt == T::pixel_format ? T::num_channels : numChannelsHelper(fmt, ct::VariadicTypedef<ARGS...>{});
    }

    constexpr uint8_t PixelFormat::numChannels() const
    {
        return numChannelsHelper(*this, AvailablePixelFormats_t<uint8_t>{});
    }
} // namespace aq

#endif // AQUILA_TYPES_PIXELS_HPP
