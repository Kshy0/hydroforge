// Packed high-precision storage. Legacy float destinations remain compatible.
// Pair overloads below prevent native-float fallback for supported operations.
#ifndef HF_EMULATED_INCLUDED
#define HF_EMULATED_INCLUDED
#pragma STDC FP_CONTRACT OFF
struct hf_hp {
    hf_float32x2::Value value;
    hf_hp() = default;
    hf_hp(float x) : value(hf_float32x2::from_float(x)) {}
    hf_hp(float hi, float lo) : value(hf_float32x2::renorm({hi, lo})) {}
    hf_hp(bool x) : value(hf_float32x2::from_float(x ? 1.0f : 0.0f)) {}
    hf_hp(int x) : value(hf_float32x2::from_long(long(x))) {}
    hf_hp(uint x) : value(hf_float32x2::from_uint(x)) {}
    hf_hp(long x) : value(hf_float32x2::from_long(x)) {}
    hf_hp(ulong x) : value(hf_float32x2::from_ulong(x)) {}
    operator float() const { return hf_float32x2::to_float(value); }
    explicit operator bool() const { return hf_float32x2::to_bool(value); }
#ifdef __METAL_VERSION__
    operator float() const device { return hf_float32x2::to_float(value); }
    explicit operator bool() const device { return hf_float32x2::to_bool(value); }
#endif
};
inline hf_hp hf_hp_from_value(hf_float32x2::Value v) { hf_hp r; r.value = v; return r; }
inline long hf_hp_to_long(hf_hp a) { return hf_float32x2::to_long(a.value); }
inline int hf_hp_to_int(hf_hp a) { return hf_float32x2::to_int(a.value); }
inline bool hf_hp_to_bool(hf_hp a) { return hf_float32x2::to_bool(a.value); }
inline hf_hp operator+(hf_hp a, hf_hp b) { return hf_hp_from_value(hf_float32x2::add(a.value, b.value)); }
inline hf_hp operator-(hf_hp a, hf_hp b) { return hf_hp_from_value(hf_float32x2::sub(a.value, b.value)); }
inline hf_hp operator*(hf_hp a, hf_hp b) { return hf_hp_from_value(hf_float32x2::mul(a.value, b.value)); }
inline hf_hp operator/(hf_hp a, hf_hp b) { return hf_hp_from_value(hf_float32x2::div(a.value, b.value)); }
inline hf_hp operator-(hf_hp a) { return hf_hp_from_value(hf_float32x2::neg(a.value)); }
inline hf_hp operator+(hf_hp a) { return a; }
inline bool operator==(hf_hp a, hf_hp b) { return hf_float32x2::eq(a.value, b.value); }
inline bool operator<(hf_hp a, hf_hp b) { return hf_float32x2::lt(a.value, b.value); }
inline bool operator<=(hf_hp a, hf_hp b) { return hf_float32x2::le(a.value, b.value); }
inline bool operator>(hf_hp a, hf_hp b) { return b < a; }
inline bool operator>=(hf_hp a, hf_hp b) { return b <= a; }
inline bool operator!=(hf_hp a, hf_hp b) { return !(a == b); }
// Both operand orders retain integer precision within pair capacity and resolve overloads in
// the presence of the legacy implicit float conversion.
template<typename T> inline hf_hp operator+(hf_hp a, T b) { return a + hf_hp(b); }
template<typename T> inline hf_hp operator+(T a, hf_hp b) { return hf_hp(a) + b; }
template<typename T> inline hf_hp operator-(hf_hp a, T b) { return a - hf_hp(b); }
template<typename T> inline hf_hp operator-(T a, hf_hp b) { return hf_hp(a) - b; }
template<typename T> inline hf_hp operator*(hf_hp a, T b) { return a * hf_hp(b); }
template<typename T> inline hf_hp operator*(T a, hf_hp b) { return hf_hp(a) * b; }
template<typename T> inline hf_hp operator/(hf_hp a, T b) { return a / hf_hp(b); }
template<typename T> inline hf_hp operator/(T a, hf_hp b) { return hf_hp(a) / b; }
template<typename T> inline bool operator==(hf_hp a, T b) { return a == hf_hp(b); }
template<typename T> inline bool operator==(T a, hf_hp b) { return hf_hp(a) == b; }
template<typename T> inline bool operator!=(hf_hp a, T b) { return a != hf_hp(b); }
template<typename T> inline bool operator!=(T a, hf_hp b) { return hf_hp(a) != b; }
template<typename T> inline bool operator<(hf_hp a, T b) { return a < hf_hp(b); }
template<typename T> inline bool operator<(T a, hf_hp b) { return hf_hp(a) < b; }
template<typename T> inline bool operator<=(hf_hp a, T b) { return a <= hf_hp(b); }
template<typename T> inline bool operator<=(T a, hf_hp b) { return hf_hp(a) <= b; }
template<typename T> inline bool operator>(hf_hp a, T b) { return a > hf_hp(b); }
template<typename T> inline bool operator>(T a, hf_hp b) { return hf_hp(a) > b; }
template<typename T> inline bool operator>=(hf_hp a, T b) { return a >= hf_hp(b); }
template<typename T> inline bool operator>=(T a, hf_hp b) { return hf_hp(a) >= b; }
inline hf_hp hf_hp_abs(hf_hp a) { return hf_hp_from_value(hf_float32x2::abs(a.value)); }
inline hf_hp hf_hp_sqrt(hf_hp a) { return hf_hp_from_value(hf_float32x2::sqrt(a.value)); }
inline hf_hp hf_hp_exp(hf_hp a) { return hf_hp_from_value(hf_float32x2::exp(a.value)); }
inline hf_hp hf_hp_log(hf_hp a) { return hf_hp_from_value(hf_float32x2::log(a.value)); }
inline hf_hp hf_hp_sin(hf_hp a) { return hf_hp_from_value(hf_float32x2::sin(a.value)); }
inline hf_hp hf_hp_cos(hf_hp a) { return hf_hp_from_value(hf_float32x2::cos(a.value)); }
inline hf_hp hf_hp_tan(hf_hp a) { return hf_hp_from_value(hf_float32x2::tan(a.value)); }
inline hf_hp hf_hp_pow(hf_hp a, hf_hp b) { return hf_hp_from_value(hf_float32x2::pow(a.value, b.value)); }
inline hf_hp hf_hp_fmod(hf_hp a, hf_hp b) { return hf_hp_from_value(hf_float32x2::fmod(a.value, b.value)); }
inline hf_hp hf_hp_mod(hf_hp a, hf_hp b) { return hf_hp_from_value(hf_float32x2::mod(a.value, b.value)); }
// Also protect handwritten kernels using unqualified Metal math names.
inline hf_hp abs(hf_hp a) { return hf_hp_abs(a); }
inline hf_hp sqrt(hf_hp a) { return hf_hp_sqrt(a); }
inline hf_hp exp(hf_hp a) { return hf_hp_exp(a); }
inline hf_hp log(hf_hp a) { return hf_hp_log(a); }
inline hf_hp sin(hf_hp a) { return hf_hp_sin(a); }
inline hf_hp cos(hf_hp a) { return hf_hp_cos(a); }
inline hf_hp tan(hf_hp a) { return hf_hp_tan(a); }
inline hf_hp pow(hf_hp a, hf_hp b) { return hf_hp_pow(a,b); }
template<typename T> inline hf_hp pow(hf_hp a, T b) { return hf_hp_pow(a,hf_hp(b)); }
template<typename T> inline hf_hp pow(T a, hf_hp b) { return hf_hp_pow(hf_hp(a),b); }
inline hf_hp fmod(hf_hp a, hf_hp b) { return hf_hp_fmod(a,b); }
template<typename T> inline hf_hp fmod(hf_hp a, T b) { return hf_hp_fmod(a,hf_hp(b)); }
template<typename T> inline hf_hp fmod(T a, hf_hp b) { return hf_hp_fmod(hf_hp(a),b); }
inline bool isnan(hf_hp a) { return hf_float32x2::isnan(a.value); }
inline bool isfinite(hf_hp a) { return hf_float32x2::isfinite(a.value); }
inline bool hydroforge_isnan(hf_hp a) { return isnan(a); }
inline hf_hp hydroforge_maximum(hf_hp a, hf_hp b) {
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    return a > b ? a : b;
}
inline hf_hp hydroforge_minimum(hf_hp a, hf_hp b) {
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    return a < b ? a : b;
}
inline hf_hp max(hf_hp a, hf_hp b) { return hydroforge_maximum(a, b); }
inline hf_hp min(hf_hp a, hf_hp b) { return hydroforge_minimum(a, b); }
template<typename T> inline hf_hp min(hf_hp a, T b) { return min(a, hf_hp(b)); }
template<typename T> inline hf_hp min(T a, hf_hp b) { return min(hf_hp(a), b); }
template<typename T> inline hf_hp max(hf_hp a, T b) { return max(a, hf_hp(b)); }
template<typename T> inline hf_hp max(T a, hf_hp b) { return max(hf_hp(a), b); }
template<typename T> inline hf_hp hydroforge_minimum(hf_hp a, T b) { return hydroforge_minimum(a, hf_hp(b)); }
template<typename T> inline hf_hp hydroforge_minimum(T a, hf_hp b) { return hydroforge_minimum(hf_hp(a), b); }
template<typename T> inline hf_hp hydroforge_maximum(hf_hp a, T b) { return hydroforge_maximum(a, hf_hp(b)); }
template<typename T> inline hf_hp hydroforge_maximum(T a, hf_hp b) { return hydroforge_maximum(hf_hp(a), b); }

inline hf_hp hydroforge_weighted_mean(hf_hp old_value, hf_hp old_weight, hf_hp value, hf_hp weight) {
    hf_hp new_weight = old_weight + weight;
    // Finite weights can overflow their sum even though both ratios and the
    // resulting mean are representable. Scale both before taking the ratio.
    if (!isfinite(new_weight) && isfinite(old_weight) && isfinite(weight)) {
        old_weight = old_weight * hf_hp(0.5f);
        weight = weight * hf_hp(0.5f);
        new_weight = old_weight + weight;
    }
    hf_hp ratio = weight / new_weight;
    hf_hp incremental = old_value + (value - old_value) * ratio;
    if (isfinite(incremental)) return incremental;
    return old_value * (old_weight / new_weight) + value * ratio;
}
#endif
