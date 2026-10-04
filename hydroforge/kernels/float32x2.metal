// Two-component FP32 arithmetic. About 48 significant bits, FP32 exponent
// range; not IEEE binary64. Requires fast-math and contraction disabled.
#ifndef HF_FLOAT32X2_INCLUDED
#define HF_FLOAT32X2_INCLUDED
#include <metal_stdlib>
namespace hf_float32x2 {
struct Value { float hi; float lo; };
inline Value from_float(float x) { return {x, 0.0f}; }
inline float to_float(Value x) { return x.hi + x.lo; }
inline Value two_sum(float a, float b) {
    float s = a + b;
    if (!metal::isfinite(s)) return {s, 0.0f};
    float v = s - a;
    return {s, (a - (s - v)) + (b - v)};
}
inline Value add(Value a, Value b) {
    Value s = two_sum(a.hi, b.hi);
    if (!metal::isfinite(s.hi)) return s;
    Value t = two_sum(a.lo, b.lo);
    Value u = two_sum(s.hi, s.lo + t.hi);
    return two_sum(u.hi, u.lo + t.lo);
}
inline Value neg(Value a) { return {-a.hi, -a.lo}; }
inline Value sub(Value a, Value b) { return add(a, neg(b)); }
inline Value mul(Value a, Value b) {
    float p = a.hi * b.hi;
    if (!metal::isfinite(p)) return {p, 0.0f};
    float e = metal::fma(a.hi, b.hi, -p);
    e = e + a.hi * b.lo;
    e = e + a.lo * b.hi;
    e = e + a.lo * b.lo;
    return two_sum(p, e);
}
inline Value div(Value a, Value b) {
    float q = a.hi / b.hi;
    if (!metal::isfinite(a.hi) || !metal::isfinite(b.hi) ||
        !metal::isfinite(q) || b.hi == 0.0f) return {q, 0.0f};
    Value residual = sub(a, mul(b, from_float(q)));
    float correction = (residual.hi + residual.lo) / b.hi;
    return two_sum(q, correction);
}
inline bool eq(Value a, Value b) { return a.hi == b.hi && a.lo == b.lo; }
inline bool lt(Value a, Value b) {
    return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
}
inline bool le(Value a, Value b) { return lt(a, b) || eq(a, b); }
inline bool isfinite(Value a) {
    return metal::isfinite(a.hi) && metal::isfinite(a.lo);
}
}
#endif
