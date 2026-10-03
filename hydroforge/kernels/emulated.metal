// Explicit packed high-precision value type used by generated Metal programs.
#ifndef HF_EMULATED_INCLUDED
#define HF_EMULATED_INCLUDED
struct hf_hp {
    hf_float32x2::Value value;
    hf_hp(float x) : value(hf_float32x2::from_float(x)) {}
    operator float() const { return hf_float32x2::to_float(value); }
    operator float() const device { return hf_float32x2::to_float(value); }
    hf_hp() = default;
};
inline hf_hp operator+(hf_hp a, hf_hp b) { hf_hp r; r.value = hf_float32x2::add(a.value, b.value); return r; }
inline hf_hp operator+(hf_hp a, float b) { return a + hf_hp(b); }
inline hf_hp operator+(float a, hf_hp b) { return hf_hp(a) + b; }
inline hf_hp operator-(hf_hp a, hf_hp b) { hf_hp r; r.value = hf_float32x2::sub(a.value, b.value); return r; }
inline hf_hp operator-(hf_hp a, float b) { return a - hf_hp(b); }
inline hf_hp operator-(float a, hf_hp b) { return hf_hp(a) - b; }
inline hf_hp operator*(hf_hp a, hf_hp b) { hf_hp r; r.value = hf_float32x2::mul(a.value, b.value); return r; }
inline hf_hp operator*(hf_hp a, float b) { return a * hf_hp(b); }
inline hf_hp operator*(float a, hf_hp b) { return hf_hp(a) * b; }
inline hf_hp operator/(hf_hp a, hf_hp b) { hf_hp r; r.value = hf_float32x2::div(a.value, b.value); return r; }
inline hf_hp operator/(hf_hp a, float b) { return a / hf_hp(b); }
inline hf_hp operator/(float a, hf_hp b) { return hf_hp(a) / b; }
inline bool operator==(hf_hp a, hf_hp b) { return hf_float32x2::eq(a.value, b.value); }
inline bool operator<(hf_hp a, hf_hp b) { return hf_float32x2::lt(a.value, b.value); }
inline bool operator<=(hf_hp a, hf_hp b) { return hf_float32x2::le(a.value, b.value); }
inline bool operator>(hf_hp a, hf_hp b) { return b < a; }
inline bool operator>=(hf_hp a, hf_hp b) { return b <= a; }
inline bool operator!=(hf_hp a, hf_hp b) { return !(a == b); }
inline bool operator==(hf_hp a, float b) { return a == hf_hp(b); }
inline bool operator==(float a, hf_hp b) { return hf_hp(a) == b; }
inline bool operator<(hf_hp a, float b) { return a < hf_hp(b); }
inline bool operator<(float a, hf_hp b) { return hf_hp(a) < b; }
inline bool operator<=(hf_hp a, float b) { return a <= hf_hp(b); }
inline bool operator<=(float a, hf_hp b) { return hf_hp(a) <= b; }
inline bool operator>(hf_hp a, float b) { return a > hf_hp(b); }
inline bool operator>(float a, hf_hp b) { return hf_hp(a) > b; }
inline bool operator>=(hf_hp a, float b) { return a >= hf_hp(b); }
inline bool operator>=(float a, hf_hp b) { return hf_hp(a) >= b; }
inline bool operator!=(hf_hp a, float b) { return a != hf_hp(b); }
inline bool operator!=(float a, hf_hp b) { return hf_hp(a) != b; }
inline hf_hp operator-(hf_hp a) { return hf_hp(0.0f) - a; }
inline hf_hp min(hf_hp a, hf_hp b) { return a < b ? a : b; }
inline hf_hp max(hf_hp a, hf_hp b) { return a > b ? a : b; }
inline bool isfinite(hf_hp a) {
    return hf_float32x2::isfinite(a.value);
}
#endif
