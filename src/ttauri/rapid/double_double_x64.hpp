

#pragma once

namespace tt {

struct dd_return_type {
    __m128 hi;
    __m128 lo;
};

[[nodiscard]] dd_return_type dd_quick_two_sum(__m128d a, __m128d b) noexcept
{
    ttlet s = _mm_add_pd(a, b);
    ttlet e = _mm_sub_pd(b, _mm_sub_pd(s, a));
    return dd_return_type{s, e};
}

[[nodiscard]] dd_return_type dd_two_sum(__m128d a, __m128d b) noexcept
{
    // clang-format off
    ttlet s = _mm_add_pd(a, b);
    ttlet v = _mm_sub_pd(s, a);
    ttlet e = _mm_add_pd(
        _mm_sub_pd(a, _mm_sub_pd(s, v)),
        _mm_sub_pd(b, v)
    );
    return dd_return_type{s, e};
    // clang-format on
}

[[nodiscard]] dd_return_type dd_two_difference(__m128d a, __m128d b) noexcept
{
    // clang-format off
    ttlet s = _mm_sub_pd(a, b);
    ttlet v = _mm_sub_pd(s, a);
    ttlet e = _mm_sub_pd(
        _mm_sub_pd(a, _mm_sub_pd(s, v)),
        _mm_add_pd(b, v)
    );
    return dd_return_type{s, e};
    // clang-format on
}

template<size_t Bits>
[[nodiscard]] dd_return_type dd_split_(__m128d a) noexcept
{
    constexpr double Scale = static_cast<double>((1_uz << Bits) + 1);

    ttlet t = _mm_mul_pd(_mm_set1_pd(Scale), a);
    ttlet hi = _mm_sub_pd(t, _mm_sub_pd(t, a));
    ttlet lo = _mm_sub_pd(a, hi);
    return dd_return_type{hi, lo};
}

[[nodiscard]] dd_return_type dd_split(__m128d a) noexcept
{
    return dd_split_<27>(a);
}


[[nodiscard]] dd_return_type dd_two_prod(__m128d a, __m128d b) noexcept
{
    ttlet p = _mm_mul_pd(a, b);
    ttlet [ahi, alo] = dd_split(a);
    ttlet [bhi, blo] = dd_split(b);

    ttlet ahi_bhi = _mm_mul_pd(ahi, bhi);
    ttlet ahi_blo = _mm_mul_pd(ahi, blo);
    ttlet alo_bhi = _mm_mul_pd(alo, bhi);
    ttlet alo_blo = _mm_mul_pd(alo, blo);

    ttlet e = _mm_add_pd(
        _mm_add_pd(
            _mm_sub_pd(ahi_bhi, p),
            _mm_add_pd(ahi_blo, alo_bhi),
        ),
        alo_blo
    );
    return dd_return_type{p, e};
}

[[nodiscard]] dd_return_type dd_two_prod(__m128d a, __m128d b) noexcept
{
    ttlet p = _mm_mul_pd(a, b);
    ttlet e = _mm_fmsub_pd(a, b, p);
    return dd_return_type{p, e};
}


class double_double {
public:


    [[nodiscard]] friend bool operator==(double_double const &lhs, double_double const &rhs) noexcept
    {
        return _mm_movemask_pd(_mm_cmpeq_pd(lhs._v, rhs._v)) == 0b11;
    }

    [[nodiscard]] friend std::strong_ordering operator<=>(double_double const &lhs, double_double const &rhs) noexcept
    {
        ttlet eq = _mm_movemask_pd(_mm_cmpeq_pd(lhs._v, rhs._v));
        if (eq == 0b11) {
            return std::strong_ordering::equivalent;
        } else {
            ttlet lt = _mm_movemask_pd(_mm_cmplt_pd(lhs._v, rhs._v));
            ttlet is_less = lt & (eq == 0b10 ? 0b01 : 0b10);
            return is_less ? std::strong_ordering::less : std::strong_ordering::greater;
        }
    }

private:
    __m128d _v;
};

}

