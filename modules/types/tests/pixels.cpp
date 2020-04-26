#include <Aquila/types/pixels.hpp>
#include <gtest/gtest.h>

template<class T>
void checkSize()
{
    static_assert(sizeof(aq::RGB<T>) == sizeof(T) * 3, "asdf");
    static_assert(sizeof(aq::BGR<T>) == sizeof(T) * 3, "asdf");
    static_assert(sizeof(aq::HSV<T>) == sizeof(T) * 3, "asdf");
    static_assert(sizeof(aq::HSL<T>) == sizeof(T) * 3, "asdf");
    static_assert(sizeof(aq::LUV<T>) == sizeof(T) * 3, "asdf");
    static_assert(sizeof(aq::ARGB<T>) == sizeof(T) * 4, "asdf");
    static_assert(sizeof(aq::RGBA<T>) == sizeof(T) * 4, "asdf");
    static_assert(sizeof(aq::ABGR<T>) == sizeof(T) * 4, "asdf");
    static_assert(sizeof(aq::BGRA<T>) == sizeof(T) * 4, "asdf");
    static_assert(sizeof(aq::GRAY<T>) == sizeof(T), "asdf");

    static_assert(std::is_trivially_copyable<aq::RGB<T>>::value, "asdf");
    static_assert(std::is_trivially_copyable<aq::BGR<T>>::value, "asdf");
    static_assert(std::is_trivially_copyable<aq::HSV<T>>::value, "asdf");
    static_assert(std::is_trivially_copyable<aq::HSV<T>>::value, "asdf");
    static_assert(std::is_trivially_copyable<aq::HSL<T>>::value, "asdf");
    static_assert(std::is_trivially_copyable<aq::LUV<T>>::value, "asdf");
    static_assert(std::is_trivially_copyable<aq::ARGB<T>>::value, "asdf");
    static_assert(std::is_trivially_copyable<aq::RGBA<T>>::value, "asdf");
    static_assert(std::is_trivially_copyable<aq::ABGR<T>>::value, "asdf");
    static_assert(std::is_trivially_copyable<aq::BGRA<T>>::value, "asdf");
    static_assert(std::is_trivially_copyable<aq::GRAY<T>>::value, "asdf");

}

TEST(pixel, size)
{
    checkSize<float>();
    checkSize<uint8_t>();
    checkSize<uint16_t>();
}

TEST(pixel, conversions)
{
    const aq::RGB<uint8_t> rgb(0,1,2);
    const auto bgr = rgb.bgr();
    EXPECT_EQ(rgb.b(), bgr.b());
    EXPECT_EQ(rgb.g(), bgr.g());
    EXPECT_EQ(rgb.r(), bgr.r());

    const auto bgra = bgr.bgra();
    EXPECT_EQ(bgra.b(), bgr.b());
    EXPECT_EQ(bgra.g(), bgr.g());
    EXPECT_EQ(bgra.r(), bgr.r());
    EXPECT_EQ(bgra.a(), 255);

    const auto abgr = bgr.abgr();
    EXPECT_EQ(abgr.b(), bgr.b());
    EXPECT_EQ(abgr.g(), bgr.g());
    EXPECT_EQ(abgr.r(), bgr.r());
    EXPECT_EQ(abgr.a(), 255);

    const auto argb = rgb.argb();
    EXPECT_EQ(argb.a(), 255);
    EXPECT_EQ(argb.r(), rgb.r());
    EXPECT_EQ(argb.g(), rgb.g());
    EXPECT_EQ(argb.b(), rgb.b());

    const auto rgba = rgb.argb();
    EXPECT_EQ(rgba.a(), 255);
    EXPECT_EQ(rgba.r(), rgb.r());
    EXPECT_EQ(rgba.g(), rgb.g());
    EXPECT_EQ(rgba.b(), rgb.b());

}
