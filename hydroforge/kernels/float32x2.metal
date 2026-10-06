// Double-FP32: a normalized unevaluated hi+lo sum, approximately 48 bits.
// This is NOT IEEE binary64: exponent range remains FP32. Low limbs lose
// precision near the FP32 subnormal range; Metal may flush subnormals to zero.
// Requires round-to-nearest FP32 operations, fast math OFF, and contraction OFF.
// Only explicit fma below is permitted (the exact multiplication residual).
#ifndef HF_FLOAT32X2_INCLUDED
#define HF_FLOAT32X2_INCLUDED
#include <metal_stdlib>
#pragma STDC FP_CONTRACT OFF
namespace hf_float32x2 {
struct Value { float hi; float lo; };
inline Value from_float(float x) { return {x, 0.0f}; }
inline float to_float(Value x) { return x.hi == 0.0f && x.lo == 0.0f ? x.hi : x.hi + x.lo; }
inline bool isnan(Value a) { return metal::isnan(a.hi) || metal::isnan(a.lo); }
inline bool isfinite(Value a) { return metal::isfinite(a.hi) && metal::isfinite(a.lo); }
inline bool zero(Value a) { return a.hi == 0.0f && a.lo == 0.0f; }
inline Value nan() { return {NAN, 0.0f}; }
inline Value inf(bool negative = false) { return {negative ? -INFINITY : INFINITY, 0.0f}; }
inline Value neg(Value a) { return {-a.hi, -a.lo}; }
inline bool signbit(Value a) { return metal::signbit(a.hi); }
inline Value abs(Value a) { return signbit(a) ? neg(a) : a; }
inline bool eq(Value a, Value b) { return a.hi == b.hi && a.lo == b.lo; }
inline bool lt(Value a, Value b) {
    if (isnan(a) || isnan(b)) return false;
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
inline bool le(Value a, Value b) { return lt(a, b) || eq(a, b); }
inline Value two_sum(float a, float b) {
    float s = a + b;
    if (!metal::isfinite(s)) return {s, 0.0f};
    float v = s - a;
    return {s, (a - (s - v)) + (b - v)};
}
inline Value renorm(Value a) { return zero(a) ? a : two_sum(a.hi, a.lo); }
inline Value scale(Value a, int e) {
    if (!isfinite(a) || zero(a)) return a;
    float h = metal::ldexp(a.hi, e);
    if (!metal::isfinite(h)) return {h, 0.0f};
    return two_sum(h, metal::ldexp(a.lo, e));
}
inline Value add(Value a, Value b) {
    if (isnan(a) || isnan(b)) return nan();
    if (zero(a) && zero(b)) return from_float(a.hi + b.hi);
    if (!isfinite(a) || !isfinite(b)) return from_float(a.hi + b.hi);
    Value s = two_sum(a.hi, b.hi);
    // Avoid losing a finite result when only the leading-limb sum overflows.
    if (!metal::isfinite(s.hi)) {
        a = scale(a, -1); b = scale(b, -1);
        s = two_sum(a.hi, b.hi);
        Value t = two_sum(a.lo, b.lo);
        Value u = two_sum(s.hi, s.lo + t.hi);
        return scale(two_sum(u.hi, u.lo + t.lo), 1);
    }
    Value t = two_sum(a.lo, b.lo);
    Value u = two_sum(s.hi, s.lo + t.hi);
    return two_sum(u.hi, u.lo + t.lo);
}
inline Value sub(Value a, Value b) { return add(a, neg(b)); }
// Inputs to this internal product have modest, normalized exponents.
inline Value mul_normal(Value a, Value b) {
    float p = a.hi * b.hi;
    float e = metal::fma(a.hi, b.hi, -p);
    e = e + a.hi * b.lo;
    e = e + a.lo * b.hi;
    e = e + a.lo * b.lo;
    return two_sum(p, e);
}
inline Value mul(Value a, Value b) {
    if (isnan(a) || isnan(b)) return nan();
    if (!isfinite(a) || !isfinite(b) || zero(a) || zero(b))
        return from_float(a.hi * b.hi);
    int ea = 0, eb = 0;
    metal::frexp(a.hi, ea); metal::frexp(b.hi, eb);
    return scale(mul_normal(scale(a, -ea), scale(b, -eb)), ea + eb);
}
inline Value div(Value a, Value b) {
    if (isnan(a) || isnan(b)) return nan();
    if (!isfinite(a) || !isfinite(b) || zero(a) || zero(b))
        return from_float(a.hi / b.hi);
    int ea = 0, eb = 0;
    metal::frexp(a.hi, ea); metal::frexp(b.hi, eb);
    a = scale(a, -ea); b = scale(b, -eb);
    float q = a.hi / b.hi;
    Value r = sub(a, mul_normal(b, from_float(q)));
    Value result = two_sum(q, (r.hi + r.lo) / b.hi);
    // A second correction removes rounding from the first residual quotient.
    r = sub(a, mul_normal(b, result));
    result = add(result, from_float((r.hi + r.lo) / b.hi));
    return scale(result, ea - eb);
}
// Integer promotion rounds into the pair representation. Every integer with
// |x| <= 2^48 is exact; arbitrary 64-bit integers may lose low bits.
inline Value from_uint(uint x) {
    return add(from_float(float(x >> 16) * 65536.0f), from_float(float(x & 65535u)));
}
inline Value from_ulong(ulong x) {
    return add(scale(from_uint(uint(x >> 32)), 32), from_uint(uint(x)));
}
inline Value from_long(long x) {
    // Avoid signed overflow when x is LONG_MIN.
    ulong magnitude = x < 0 ? ulong(-(x + 1)) + 1ul : ulong(x);
    Value v = from_ulong(magnitude);
    return x < 0 ? neg(v) : v;
}
inline Value floor(Value a) {
    if (!isfinite(a) || zero(a)) return a;
    float h = metal::floor(a.hi);
    return h == a.hi ? add(from_float(h), from_float(metal::floor(a.lo))) : from_float(h);
}
inline Value ceil(Value a) { return neg(floor(neg(a))); }
inline Value trunc(Value a) { return signbit(a) ? ceil(a) : floor(a); }
// Explicit, deterministic casts: truncate toward zero, saturate out-of-range,
// and map NaN to zero. No intermediate float32 rounding of the logical value.
inline long to_long(Value a) {
    if (isnan(a)) return 0l;
    const long maximum = 0x7fffffffffffffffl;
    const long minimum = (-0x7fffffffffffffffl - 1l);
    if (le({0x1p63f, -1.0f}, a)) return maximum;
    if (le(a, {-0x1p63f, 0.0f})) return minimum;
    a = trunc(a);
    if (a.hi >= 0x1p63f)
        return (long(a.hi - 0x1p62f) + long(a.lo)) + 0x4000000000000000l;
    return long(a.hi) + long(a.lo);
}
inline int to_int(Value a) {
    long v = to_long(a);
    if (v > 2147483647l) return 2147483647;
    if (v < -2147483648l) return (-2147483647 - 1);
    return int(v);
}
inline bool to_bool(Value a) { return !zero(a); }
constant Value LN2 = {0x1.62e4300000000p-1f, -0x1.05c6100000000p-29f};
constant Value PIO2 = {0x1.921fb60000000p+0f, -0x1.777a5c0000000p-25f};
constant Value PIO4 = {0x1.921fb60000000p-1f, -0x1.777a5c0000000p-26f};
// Cody-Waite splits: n * LN2_1/LN2_2 is exact for |n| < 2^8 and
// q * PIO2_1..PIO2_3 for |q| < 2^10; the pair tails carry the remainder.
constant float LN2_1 = 0x1.62e4p-1f, LN2_2 = 0x1.7f7cp-20f;
constant Value LN2_3 = {0x1.1cf79ap-36f, 0x1.793c76p-61f};
constant float PIO2_1 = 0x1.9218p+0f, PIO2_2 = 0x1.ed5p-14f, PIO2_3 = 0x1.10bp-30f;
constant Value PIO2_4 = {0x1.184698p-44f, 0x1.3198a2p-69f};
// Pair reciprocals 1/n at index n (index 0 unused); series terms multiply by them.
constant Value RECIPROCAL[28] = {
    {0x0.0p+0f, 0x0.0p+0f}, {0x1.0000000000000p+0f, 0x0.0p+0f},
    {0x1.0000000000000p-1f, 0x0.0p+0f}, {0x1.5555560000000p-2f, -0x1.5555560000000p-27f},
    {0x1.0000000000000p-2f, 0x0.0p+0f}, {0x1.99999a0000000p-3f, -0x1.99999a0000000p-29f},
    {0x1.5555560000000p-3f, -0x1.5555560000000p-28f}, {0x1.24924a0000000p-3f, -0x1.b6db6e0000000p-28f},
    {0x1.0000000000000p-3f, 0x0.0p+0f}, {0x1.c71c720000000p-4f, -0x1.c71c720000000p-31f},
    {0x1.99999a0000000p-4f, -0x1.99999a0000000p-30f}, {0x1.745d180000000p-4f, -0x1.745d180000000p-29f},
    {0x1.5555560000000p-4f, -0x1.5555560000000p-29f}, {0x1.3b13b20000000p-4f, -0x1.89d89e0000000p-29f},
    {0x1.24924a0000000p-4f, -0x1.b6db6e0000000p-29f}, {0x1.1111120000000p-4f, -0x1.ddddde0000000p-29f},
    {0x1.0000000000000p-4f, 0x0.0p+0f}, {0x1.e1e1e20000000p-5f, -0x1.e1e1e20000000p-33f},
    {0x1.c71c720000000p-5f, -0x1.c71c720000000p-32f}, {0x1.af286c0000000p-5f, -0x1.af286c0000000p-32f},
    {0x1.99999a0000000p-5f, -0x1.99999a0000000p-31f}, {0x1.8618620000000p-5f, -0x1.e79e7a0000000p-31f},
    {0x1.745d180000000p-5f, -0x1.745d180000000p-30f}, {0x1.642c860000000p-5f, -0x1.bd37a60000000p-31f},
    {0x1.5555560000000p-5f, -0x1.5555560000000p-30f}, {0x1.47ae140000000p-5f, 0x1.eb851e0000000p-31f},
    {0x1.3b13b20000000p-5f, -0x1.89d89e0000000p-30f}, {0x1.2f684c0000000p-5f, -0x1.2f684c0000000p-32f}
};
// 1/((2k)(2k+1)) at index k: the ratio of consecutive sine series terms.
constant Value SIN_STEP[11] = {
    {0x0.0p+0f, 0x0.0p+0f}, {0x1.5555560000000p-3f, -0x1.5555560000000p-28f},
    {0x1.99999a0000000p-5f, -0x1.99999a0000000p-31f}, {0x1.8618620000000p-6f, -0x1.e79e7a0000000p-32f},
    {0x1.c71c720000000p-7f, -0x1.c71c720000000p-34f}, {0x1.29e4120000000p-7f, 0x1.3c82540000000p-32f},
    {0x1.a41a420000000p-8f, -0x1.6f96fa0000000p-34f}, {0x1.3813820000000p-8f, -0x1.8fd8fe0000000p-33f},
    {0x1.e1e1e20000000p-9f, -0x1.e1e1e20000000p-37f}, {0x1.7f40600000000p-9f, -0x1.7f40600000000p-36f},
    {0x1.3813820000000p-9f, -0x1.8fd8fe0000000p-34f}
};
// 1/((2k-1)(2k)) at index k: the ratio of consecutive cosine series terms.
constant Value COS_STEP[11] = {
    {0x0.0p+0f, 0x0.0p+0f}, {0x1.0000000000000p-1f, 0x0.0p+0f},
    {0x1.5555560000000p-4f, -0x1.5555560000000p-29f}, {0x1.1111120000000p-5f, -0x1.ddddde0000000p-30f},
    {0x1.24924a0000000p-6f, -0x1.b6db6e0000000p-31f}, {0x1.6c16c20000000p-7f, -0x1.27d27e0000000p-32f},
    {0x1.f07c200000000p-8f, -0x1.f07c200000000p-33f}, {0x1.6816820000000p-8f, -0x1.2fd2fe0000000p-33f},
    {0x1.1111120000000p-8f, -0x1.ddddde0000000p-33f}, {0x1.ac57020000000p-9f, -0x1.4ea3fa0000000p-35f},
    {0x1.58ed240000000p-9f, -0x1.efd4e20000000p-34f}
};

inline Value sqrt(Value a) {
    if (isnan(a)) return nan();
    if (zero(a)) return a; // Preserve sqrt(-0).
    if (signbit(a)) return nan();
    if (!isfinite(a)) return a;
    int e = 0; metal::frexp(a.hi, e);
    int k = e / 2;
    Value m = scale(a, -2 * k);
    Value q = from_float(metal::sqrt(m.hi));
    for (int i = 0; i < 2; ++i)
        q = add(q, div(sub(m, mul_normal(q, q)), scale(q, 1)));
    return scale(q, k);
}
inline Value exp(Value a) {
    if (isnan(a)) return nan();
    if (!isfinite(a)) return signbit(a) ? from_float(0.0f) : a;
    // Guards bound the integer reduction, beyond FP32 overflow/underflow.
    if (lt(from_float(90.0f), a)) return inf();
    if (lt(a, from_float(-105.0f))) return from_float(0.0f);
    float n = metal::rint(a.hi * 1.4426950408889634f);
    Value r = sub(a, from_float(n * LN2_1));
    r = sub(r, from_float(n * LN2_2));
    r = sub(r, mul(from_float(n), LN2_3));
    Value term = from_float(1.0f), result = term;
    // |r| <= 0.347; 18 Taylor terms leave < 2^-83 truncation error.
    for (int k = 1; k <= 18; ++k) {
        term = mul_normal(mul_normal(term, r), RECIPROCAL[k]);
        result = add(result, term);
    }
    return scale(result, int(n));
}
inline Value log(Value a) {
    if (isnan(a)) return nan();
    if (zero(a)) return inf(true);
    if (signbit(a)) return nan();
    if (!isfinite(a)) return a;
    int e = 0; metal::frexp(a.hi, e);
    Value m = scale(a, -e);
    if (m.hi < 0.7071067811865476f) { m = scale(m, 1); --e; }
    Value z = div(sub(m, from_float(1.0f)), add(m, from_float(1.0f)));
    Value z2 = mul_normal(z, z), term = z, result = z;
    // atanh series: |z| <= 0.172; the tail after z^27 is < 2^-76.
    for (int k = 3; k <= 27; k += 2) {
        term = mul_normal(term, z2);
        result = add(result, mul_normal(term, RECIPROCAL[k]));
    }
    return add(scale(result, 1), mul(from_long(long(e)), LN2));
}
inline void bits_from_float(float a, thread uint* out) {
    for (int i = 0; i < 18; ++i) out[i] = 0u;
    uint bits = as_type<uint>(a) & 0x7fffffffu;
    uint exponent = bits >> 23;
    uint mantissa = bits & 0x7fffffu;
    if (exponent) mantissa |= 0x800000u;
    int shift = exponent ? int(exponent) - 1 : 0;
    int digit = shift / 16;
    ulong shifted = ulong(mantissa) << (shift % 16);
    for (int i = 0; i < 3 && digit + i < 18; ++i) {
        out[digit + i] = uint(shifted & 65535ul); shifted >>= 16;
    }
}
inline int bits_compare(thread const uint* a, thread const uint* b) {
    for (int i = 17; i >= 0; --i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}
inline void bits_subtract(thread uint* a, thread const uint* b) {
    uint borrow = 0u;
    for (int i = 0; i < 18; ++i) {
        uint subtrahend = b[i] + borrow;
        borrow = a[i] < subtrahend ? 1u : 0u;
        a[i] = (a[i] - subtrahend) & 65535u;
    }
}
inline void bits_from_value(Value a, thread uint* out) {
    uint low[18]; bits_from_float(a.hi, out); bits_from_float(a.lo, low);
    if (metal::signbit(a.lo)) bits_subtract(out, low);
    else {
        uint carry = 0u;
        for (int i = 0; i < 18; ++i) {
            uint digit = out[i] + low[i] + carry;
            out[i] = digit & 65535u; carry = digit >> 16;
        }
    }
}
inline int bits_top(thread const uint* a) {
    for (int i = 17; i >= 0; --i) {
        if (a[i]) {
            uint v = a[i]; int bit = 0;
            while (v >>= 1) ++bit;
            return 16 * i + bit;
        }
    }
    return -1;
}
// Payne-Hanek reduction. The 256-bit floor(2/pi * 2^256) is stored
// least-significant base-2^16 digit first. Fixed-point limb multiplication
// is exact in ulong. This covers EVERY finite FP32 exponent; no
// large float-to-integer conversion and no native-float trig fallback.
constant uint TWO_OVER_PI[16] = {0xc561u, 0xdebbu, 0x63abu, 0xfe51u, 0x9041u, 0x3c43u, 0x9599u, 0xdb62u, 0xddc0u, 0xf534u, 0x57d1u, 0xfc27u, 0x1529u, 0x4e44u, 0x836eu, 0xa2f9u};
inline uint ph_bit(thread const uint* digits, int bit) {
    return bit < 0 || bit >= 544 ? 0u : (digits[bit / 16] >> (bit % 16)) & 1u;
}
inline uint ph_chunk(thread const uint* digits, int top) {
    uint result = 0u;
    for (int j = 0; j < 24; ++j) result = (result << 1) | ph_bit(digits, top - j);
    return result;
}
struct Reduced { Value fraction; int quadrant; };
inline Reduced reduce(Value a) {
    // Multiply the EXACT sum of both input limbs, not two independently
    // rounded reductions. That distinction retains relative precision close
    // to zeros of sin/cos and poles of tan after hi/lo cancellation.
    uint magnitude[18], product[34];
    bits_from_value(abs(a), magnitude);
    ulong carry = 0ul;
    for (int k = 0; k < 34; ++k) {
        ulong digit = carry;
        for (int i = 0; i < 18; ++i) {
            int j = k - i;
            if (j >= 0 && j < 16) digit += ulong(magnitude[i]) * ulong(TWO_OVER_PI[j]);
        }
        product[k] = uint(digit & 65535ul); carry = digit >> 16;
    }
    // Input integers use 2^-149 units, and the table uses 2^-256 units.
    const int point = 405;
    int quadrant = int(ph_bit(product, point) | (ph_bit(product, point + 1) << 1));
    bool negative_fraction = ph_bit(product, point - 1) != 0u;
    if (negative_fraction) quadrant = (quadrant + 1) & 3;
    // Keep |fraction| exactly, taking its complement when rounded upward.
    for (int i = 0; i < 26; ++i)
        if (negative_fraction) product[i] = 65535u - product[i];
    product[25] &= 31u;
    if (negative_fraction) {
        uint increment = 1u;
        for (int i = 0; i < 26; ++i) {
            uint v = product[i] + increment;
            product[i] = v & 65535u; increment = v >> 16;
        }
    }
    for (int i = 26; i < 34; ++i) product[i] = 0u;
    int top = point - 1;
    while (top >= 0 && ph_bit(product, top) == 0u) --top;
    Value fraction = from_float(0.0f);
    if (top >= 0) {
        for (int k = 0; k < 3; ++k)
            fraction = add(fraction, scale(from_float(float(ph_chunk(product, top - 24*k))), top - 23 - 24*k - point));
    }
    if (negative_fraction) fraction = neg(fraction);
    if (signbit(a)) { fraction = neg(fraction); quadrant = (-quadrant) & 3; }
    return {mul_normal(fraction, PIO2), quadrant};
}
// Cody-Waite reduction of moderate arguments. Its pair subtractions lose
// relative precision only near a zero of the reduced argument, which the
// exact Payne-Hanek reduction then handles.
inline bool reduce_moderate(Value a, thread Reduced& reduced) {
    if (!(metal::fabs(a.hi) <= 1024.0f)) return false;
    float q = metal::rint(a.hi * 0x1.45f306p-1f);  // 2/pi
    Value r = sub(a, from_float(q * PIO2_1));
    r = sub(r, from_float(q * PIO2_2));
    r = sub(r, from_float(q * PIO2_3));
    r = sub(r, mul(from_float(q), PIO2_4));
    if (metal::fabs(r.hi) < 0x1p-10f) return false;
    reduced = Reduced{r, int(q) & 3};
    return true;
}
inline void sincos(Value a, thread Value& s, thread Value& c) {
    if (!isfinite(a)) { s = nan(); c = nan(); return; }
    if (zero(a)) { s = a; c = from_float(1.0f); return; }
    Reduced reduced = {a, 0};
    if (!le(abs(a), PIO4) && !reduce_moderate(a, reduced)) reduced = reduce(a);
    Value x = reduced.fraction, x2 = neg(mul_normal(x, x));
    Value st = x, ct = from_float(1.0f);
    s = st; c = ct;
    // |x| <= pi/4. Terms through sin(x)^19/cos(x)^20 bound truncation
    // below 2^-72; pair rounding, not native trig accuracy, dominates.
    for (int k = 1; k <= 10; ++k) {
        st = mul_normal(mul_normal(st, x2), SIN_STEP[k]);
        ct = mul_normal(mul_normal(ct, x2), COS_STEP[k]);
        s = add(s, st); c = add(c, ct);
    }
    Value old_s = s;
    if (reduced.quadrant == 1) { s = c; c = neg(old_s); }
    else if (reduced.quadrant == 2) { s = neg(s); c = neg(c); }
    else if (reduced.quadrant == 3) { s = neg(c); c = old_s; }
}
inline Value sin(Value a) { Value s, c; sincos(a, s, c); return s; }
inline Value cos(Value a) { Value s, c; sincos(a, s, c); return c; }
inline Value tan(Value a) { Value s, c; sincos(a, s, c); return div(s, c); }
// Exact fixed-point remainder, in units of 2^-149. Each pair is an exact
// sum of two FP32 numbers and therefore fits in 277 magnitude bits. Ordinary
// pair long division would repeatedly round, giving incorrect remainders when
// a/b is huge; these 18 base-2^16 limbs preserve every input bit instead.
inline Value fmod(Value a, Value b) {
    if (isnan(a) || isnan(b) || !isfinite(a) || zero(b)) return nan();
    if (!isfinite(b) || zero(a)) return a;
    Value x = abs(a), y = abs(b);
    if (lt(x, y)) return a;
    uint numerator[18], denominator[18], shifted[18];
    bits_from_value(x, numerator); bits_from_value(y, denominator);
    int shift = bits_top(numerator) - bits_top(denominator);
    for (int i = 0; i < 18; ++i) shifted[i] = 0u;
    for (int i = 0; i < 18; ++i) {
        int target = i + shift / 16;
        uint digit = denominator[i] << (shift % 16);
        if (target < 18) shifted[target] |= digit & 65535u;
        if (target + 1 < 18) shifted[target + 1] |= digit >> 16;
    }
    for (int k = shift; k >= 0; --k) {
        if (bits_compare(numerator, shifted) >= 0) bits_subtract(numerator, shifted);
        for (int i = 0; i < 18; ++i)
            shifted[i] = (shifted[i] >> 1) | (i < 17 ? (shifted[i+1] & 1u) << 15 : 0u);
    }
    Value remainder = from_float(0.0f);
    for (int i = 17; i >= 0; --i)
        if (numerator[i]) remainder = add(remainder, scale(from_float(float(numerator[i])), 16*i-149));
    return signbit(a) ? neg(remainder) : remainder;
}
inline Value mod(Value a, Value b) {
    Value r = fmod(a, b);
    if (isnan(r)) return r;
    if (zero(r)) return {metal::copysign(0.0f, b.hi), 0.0f};
    return signbit(r) != signbit(b) ? add(r, b) : r;
}
inline Value pow(Value a, Value b) {
    const Value one = {1.0f, 0.0f};
    if (zero(b) || eq(a, one)) return one;
    if (isnan(a) || isnan(b)) return nan();
    Value magnitude = abs(a);
    if (!isfinite(b)) {
        if (eq(magnitude, one)) return one;
        bool grows = lt(one, magnitude) != signbit(b);
        return grows ? inf() : from_float(0.0f);
    }
    bool integral = eq(b, trunc(b));
    bool odd = integral && !zero(fmod(abs(b), from_float(2.0f)));
    bool negative = signbit(a) && odd;
    if (zero(a) || !isfinite(a)) {
        bool infinite = zero(a) == signbit(b);
        return infinite ? inf(negative) : from_float(negative ? -0.0f : 0.0f);
    }
    if (signbit(a) && !integral) return nan();
    Value result;
    if (integral && le(abs(b), from_float(256.0f))) {
        uint n = uint(to_int(abs(b)));
        Value base = signbit(b) ? div(one, magnitude) : magnitude;
        result = one;
        while (n != 0u) {
            if (n & 1u) result = mul(result, base);
            n >>= 1;
            if (n) base = mul(base, base);
        }
    } else result = exp(mul(b, log(magnitude)));
    return negative ? neg(result) : result;
}
} // namespace hf_float32x2
#endif
