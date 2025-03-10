// Copyright Take Vos 2021.
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or copy at https://www.boost.org/LICENSE_1_0.txt)

/** @file image/sfloat_rgba32.hpp Defines the sfloat_rgba32.
 * @ingroup image
 */

#pragma once

#include "../SIMD/module.hpp"
#include "../geometry/module.hpp"
#include "../color/module.hpp"
#include <algorithm>

namespace hi::inline v1 {

/** 4 x float32 pixel format.
 *
 * @ingroup image
 */
class sfloat_rgba32 {
    // Red, Green, Blue, Alpha in binary32 (native endian).
    std::array<float, 4> v;

public:
    sfloat_rgba32() = default;
    sfloat_rgba32(sfloat_rgba32 const &rhs) noexcept = default;
    sfloat_rgba32(sfloat_rgba32 &&rhs) noexcept = default;
    sfloat_rgba32 &operator=(sfloat_rgba32 const &rhs) noexcept = default;
    sfloat_rgba32 &operator=(sfloat_rgba32 &&rhs) noexcept = default;

    sfloat_rgba32(f32x4 const &rhs) noexcept : v(static_cast<std::array<float, 4>>(rhs)) {}

    sfloat_rgba32 &operator=(f32x4 const &rhs) noexcept
    {
        v = static_cast<std::array<float, 4>>(rhs);
        return *this;
    }

    operator f32x4() const noexcept
    {
        return f32x4{v};
    }

    sfloat_rgba32(point3 const &rhs) noexcept : sfloat_rgba32(f32x4{rhs}) {}

    sfloat_rgba32(aarectangle const &rhs) noexcept : sfloat_rgba32(f32x4{rhs}) {}

    sfloat_rgba32(corner_radii const &rhs) noexcept : sfloat_rgba32(f32x4{rhs}) {}

    operator aarectangle() const noexcept
    {
        return aarectangle{f32x4{v}};
    }

    [[nodiscard]] friend bool operator==(sfloat_rgba32 const &lhs, sfloat_rgba32 const &rhs) noexcept = default;
};

} // namespace hi::inline v1
