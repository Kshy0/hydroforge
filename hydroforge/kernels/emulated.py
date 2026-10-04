"""Experimental explicitly encoded high-precision Metal tensors.

Logical values are float64; physical storage is an int64 bit carrier. Only
implemented operations are accepted: there is no arithmetic CPU fallback.
"""

from contextlib import contextmanager
from functools import cache
from pathlib import Path

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

from hydroforge.kernels import float32x2


@cache
def source() -> str:
    return (
        float32x2.source()
        + "\nusing namespace metal;\n"
        + Path(__file__).with_suffix(".metal").read_text()
    )


class EmulatedTensor(torch.Tensor):
    """Tensor metadata plus owned or aliased encoded device storage."""

    encoding = "float32x2"

    __torch_function__ = torch._C._disabled_torch_function_impl

    @staticmethod
    def __new__(cls, carrier: torch.Tensor):
        if carrier.dtype != torch.int64:
            raise TypeError("encoded Metal storage requires an int64 carrier")
        result = torch.Tensor._make_wrapper_subclass(
            cls,
            carrier.shape,
            strides=carrier.stride(),
            storage_offset=carrier.storage_offset(),
            dtype=torch.float64,
            device=carrier.device,
            requires_grad=False,
        )
        result.carrier = carrier
        return result

    def data_ptr(self):
        return self.carrier.data_ptr()

    def untyped_storage(self):
        return self.carrier.untyped_storage()

    @classmethod
    def encode(cls, values: torch.Tensor, device):
        bits = float32x2.encode(values.double().cpu(), device)
        return cls(bits.view(torch.int64).squeeze(-1))

    def decode(self):
        bits = self.carrier.detach().cpu().contiguous()
        return float32x2.decode(bits.unsqueeze(-1).view(torch.float32))

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if "out" in kwargs:
            raise NotImplementedError(
                "encoded tensor operations do not support out= buffers"
            )
        tensors = []

        def unwrap(x):
            if isinstance(x, cls):
                tensors.append(x)
                return x.carrier
            return x

        raw_args = tree_map(unwrap, args)
        raw_kwargs = tree_map(unwrap, kwargs)
        first = tensors[0]
        op = func._schema.name.removeprefix("aten::")
        if op in {
            "detach",
            "clone",
            "view",
            "reshape",
            "_unsafe_view",
            "slice",
            "select",
            "unsqueeze",
            "squeeze",
            "expand",
            "permute",
            "transpose",
            "alias",
            "index_select",
            "index",
        }:
            if "dtype" in raw_kwargs or func._overloadname == "dtype":
                raise TypeError("encoded tensor views cannot change dtype")
            return tree_map(
                lambda x: cls(x) if isinstance(x, torch.Tensor) else x,
                func(*raw_args, **raw_kwargs),
            )
        if op == "where":
            if not isinstance(args[1], cls) or not isinstance(args[2], cls):
                raise TypeError("encoded where requires two encoded value tensors")
            return cls(func(*raw_args, **raw_kwargs))
        if op == "index_put_":
            if not isinstance(args[2], cls):
                raise TypeError("encoded indexed assignment requires encoded values")
            if len(args) > 3 and args[3] or kwargs.get("accumulate", False):
                raise NotImplementedError(
                    "encoded indexed assignment cannot accumulate bits"
                )
            func(*raw_args, **raw_kwargs)
            return first
        if op == "_to_copy":
            device = torch.device(kwargs.get("device", first.device))
            dtype = kwargs.get("dtype", torch.float64)
            if device.type == "cpu":
                return first.decode().to(dtype=dtype, copy=True)
            if device.type == "mps" and dtype == torch.float64:
                return cls(first.carrier.to(device=device, copy=True))
            if device.type == "mps" and dtype == torch.float32:
                return _pointwise("convert", first)
            raise TypeError(
                "encoded tensors support only CPU export or MPS float32/float64"
            )
        if op in {"zero_", "fill_"}:
            value = 0.0 if op == "zero_" else args[1]
            encoded = cls.encode(torch.tensor(value, dtype=torch.float64), "cpu")
            first.carrier.fill_(encoded.carrier.item())
            return first
        if op == "copy_":
            if not isinstance(args[0], cls):
                raise NotImplementedError(
                    "copy encoded values using an explicit cpu() or float() conversion"
                )
            other = args[1]
            if isinstance(other, cls):
                first.carrier.copy_(other.carrier, **kwargs)
            elif other.device.type == "cpu":
                first.carrier.copy_(cls.encode(other, "cpu").carrier, **kwargs)
            else:
                raise NotImplementedError(
                    "copy from an ordinary device tensor to encoded storage"
                )
            return first
        if op in {"empty_like", "zeros_like", "ones_like"}:
            dtype = kwargs.get("dtype", torch.float64)
            device = torch.device(kwargs.get("device", first.device))
            if dtype != torch.float64 or device.type == "cpu":
                return func(
                    first.carrier, **{**kwargs, "dtype": dtype, "device": device}
                )
            carrier = torch.empty_like(
                first.carrier, **{**kwargs, "dtype": torch.int64}
            )
            result = cls(carrier)
            if op != "empty_like":
                result.fill_(0.0 if op == "zeros_like" else 1.0)
            return result
        if op in {
            "add",
            "sub",
            "mul",
            "div",
            "lt",
            "le",
            "eq",
            "ne",
            "gt",
            "ge",
            "abs",
        }:
            if kwargs:
                if kwargs != {"alpha": 1}:
                    raise NotImplementedError(f"encoded arithmetic arguments: {kwargs}")
            return _pointwise(op, *args)
        # torch.isfinite decomposes into abs/ne/eq; comparisons stay on Metal.
        raise NotImplementedError(
            f"encoded Metal tensor operation is not implemented: {func}"
        )


@cache
def _program(op, right_scalar):
    from hydroforge.kernels.metal import MetalArgument, MetalProgram

    expressions = {
        "add": "a + b",
        "sub": "a - b",
        "mul": "a * b",
        "div": "a / b",
        "lt": "a < b",
        "le": "a <= b",
        "eq": "a == b",
        "ne": "a != b",
        "gt": "a > b",
        "ge": "a >= b",
        "abs": "a == hf_hp(0.0f) ? hf_hp(0.0f) : (a < hf_hp(0.0f) ? -a : a)",
        "convert": "float(a)",
    }
    boolean = op in {"lt", "le", "eq", "ne", "gt", "ge"}
    result_type = "uchar" if boolean else "float" if op == "convert" else "hf_hp"
    right_type = "constant long" if right_scalar else "device const hf_hp"
    if not right_scalar:
        right_value = "hf_hp b = args.y[i];"
    else:
        right_value = "hf_hp b; ulong bits = as_type<ulong>(*args.y); b.value.hi = as_type<float>(uint(bits)); b.value.lo = as_type<float>(uint(bits >> 32));"
    shader = (
        source()
        + f"""
struct Args {{ device const hf_hp* x [[id(0)]]; {right_type}* y [[id(1)]];
 device {result_type}* z [[id(2)]]; constant long* n [[id(3)]]; }};
kernel void pointwise(constant Args& args [[buffer(0)]], uint i [[thread_position_in_grid]]) {{
 if (i >= *args.n) return;
 hf_hp a = args.x[i];
 {right_value}
 args.z[i] = {expressions[op]};
}}
"""
    )
    return MetalProgram(
        shader,
        "pointwise",
        (
            MetalArgument("x", "read", "long"),
            MetalArgument(
                "y",
                None if right_scalar else "read",
                "index" if right_scalar else "long",
            ),
            MetalArgument(
                "z", "write", result_type if result_type != "hf_hp" else "long"
            ),
            MetalArgument("n", None, "index"),
        ),
        extent=("n",),
    )


def _pointwise(op, a, b=0.0):
    if not isinstance(a, EmulatedTensor):
        raise NotImplementedError("encoded arithmetic requires tensor on the left")
    if a.device.type != "mps":
        raise NotImplementedError("encoded arithmetic requires actual Metal hardware")
    scalar = not isinstance(b, torch.Tensor)
    if not scalar:
        if not isinstance(b, EmulatedTensor):
            raise TypeError("encoded arithmetic requires encoded tensor operands")
        left, right = torch.broadcast_tensors(a.carrier, b.carrier)
        a = EmulatedTensor(left)
        b = EmulatedTensor(right.contiguous())
    a = a.contiguous()
    boolean = op in {"lt", "le", "eq", "ne", "gt", "ge"}
    dtype = torch.bool if boolean else torch.float32 if op == "convert" else torch.int64
    out = torch.empty(a.shape, dtype=dtype, device=a.device)
    if a.numel() == 0:
        return out if boolean or op == "convert" else EmulatedTensor(out)
    launch = _program(op, scalar).specialize(
        {
            "x": a.carrier,
            "y": EmulatedTensor.encode(
                torch.tensor(b, dtype=torch.float64), "cpu"
            ).carrier.item()
            if scalar
            else b.carrier,
            "z": out,
            "n": a.numel(),
        },
        256,
    )
    try:
        launch()
    finally:
        close = getattr(launch, "close", None)
        if callable(close):
            close()
    return out if boolean or op == "convert" else EmulatedTensor(out)


class _FactoryMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if args and isinstance(args[0], EmulatedTensor):
            return func(*args, **kwargs)
        device = kwargs.get("device")
        if device is None and args and isinstance(args[0], torch.Tensor):
            device = args[0].device
        if device is not None and torch.device(device).type == "mps":
            dtype = kwargs.get("dtype")
            if dtype is None and args and isinstance(args[0], torch.Tensor):
                dtype = args[0].dtype
            if dtype == torch.float64:
                if func._schema.name not in {
                    "aten::empty",
                    "aten::empty_strided",
                    "aten::full",
                    "aten::zeros",
                    "aten::ones",
                    "aten::_to_copy",
                    "aten::zeros_like",
                    "aten::empty_like",
                    "aten::ones_like",
                }:
                    return func(*args, **kwargs)
                cpu_kwargs = {**kwargs, "device": torch.device("cpu")}
                cpu_args = args
                if (
                    func._schema.name == "aten::_to_copy"
                    and args[0].device.type == "mps"
                ):
                    # Transfer in the source dtype before CPU conversion; MPS
                    # cannot combine its copy with a native float64 conversion.
                    cpu_args = (args[0].cpu(), *args[1:])
                value = func(*cpu_args, **cpu_kwargs)
                # Empty allocations must not encode arbitrary NaNs or out-of-range bits.
                if "empty" in func._schema.name:
                    value.zero_()
                return EmulatedTensor.encode(value, device)
        return func(*args, **kwargs)


@contextmanager
def factories(encoding):
    """Scope MPS float64 allocation to one model's cold initialization."""
    if encoding == "native":
        yield
        return
    if encoding != "float32x2":
        raise ValueError(f"unsupported Metal storage representation: {encoding!r}")
    with _FactoryMode():
        yield
