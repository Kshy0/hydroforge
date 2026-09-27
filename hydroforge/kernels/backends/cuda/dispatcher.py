"""Declarative CUDA extension namespace and canonical launch adapter."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Self

import torch
from pydantic import (
    AfterValidator,
    Field,
    FiniteFloat,
    PrivateAttr,
    ValidationInfo,
    model_validator,
)

from hydroforge.contracts.kernels import (
    BackendLoweringSpec,
    BufferDTypeABI,
    KernelSpec,
    _host_scalar_is_valid,
)
from hydroforge.contracts.naming import Identifier
from hydroforge.contracts.validation import (
    FrozenMapping,
    HydroForgeModel,
    _immutable_dict,
)
from hydroforge.kernels.backends.cuda import rtc
from hydroforge.kernels.backends.cuda.launch import (
    CudaKernel,
    CudaStep,
    CudaWorkspace,
    allocate_workspace,
    compile_steps,
    render_steps,
)
from hydroforge.kernels.backends.cuda.spec import CudaExtensionSpec
from hydroforge.kernels.context import (
    active_kernel_spec,
    registry_factory,
)

CudaProjectionValue = bool | int | FiniteFloat | None
LaunchPlan = Callable[..., Sequence[rtc.LaunchStep]]


def _named_callable(value: Any) -> Callable | None:
    if value is None:
        return None
    if not callable(value) or not getattr(value, "__name__", "").isidentifier():
        raise ValueError("CUDA route launch/check must be a named callable")
    return value


class CudaNativeProjection(HydroForgeModel):
    """Semantic preconditions for canonical values a launch plan omits."""

    fixed: FrozenMapping[Identifier, CudaProjectionValue] = Field(default_factory=dict)

    def _validate(self, values: Mapping[str, Any], *, kernel: str) -> None:
        mismatched = {
            name: (values[name], expected)
            for name, expected in self.fixed.items()
            if (
                type(values[name]) is not type(expected)
                or values[name] != expected
                or (type(expected) is float and values[name].hex() != expected.hex())
            )
        }
        if mismatched:
            detail = ", ".join(
                f"{name}={observed!r}, required={expected!r}"
                for name, (observed, expected) in sorted(mismatched.items())
            )
            raise ValueError(
                f"{kernel}: CUDA native projection precondition failed: {detail}"
            )


class CudaRoute(HydroForgeModel):
    """One kernel entry over a device source owned by a CUDA extension group.

    A route declares its launches or supplies a ``launch`` plan:

    - ``kernel="k_step<{h_ptr}>"`` declares one hand-written kernel that takes
      the canonical values over a 1-D grid of ``size_key``; it is shorthand for
      ``steps=(CudaKernel(kernel=...),)``.
    - ``steps`` lists :class:`CudaKernel` launches (hand-written kernels or
      device functions with generated entries) and :class:`CudaFill`
      operations, run in stream order over the named ``workspace`` buffers.
    - ``launch`` is a plan that receives the canonical values it names (plus
      ``BLOCK_SIZE``) and returns the launches and stream-ordered tensor
      operations of one call. It runs once per specialization.

    ``ignore`` lists canonical values no declared launch takes. ``check`` is an
    optional callable, named like a plan's parameters, that raises on canonical
    inputs the kernels cannot serve; it runs once per specialization.
    """

    extension: Identifier
    spec: KernelSpec
    launch: Annotated[Any, AfterValidator(_named_callable)] = None
    kernel: str | None = Field(default=None, min_length=1)
    steps: tuple[CudaStep, ...] = ()
    workspace: FrozenMapping[Identifier, CudaWorkspace] = Field(default_factory=dict)
    ignore: tuple[Identifier, ...] = ()
    check: Annotated[Any, AfterValidator(_named_callable)] = None
    projection: CudaNativeProjection | None = None

    @model_validator(mode="after")
    def _one_entry(self) -> Self:
        if (
            sum((self.launch is not None, self.kernel is not None, bool(self.steps)))
            != 1
        ):
            raise ValueError(
                "CUDA route requires exactly one of kernel, steps or launch"
            )
        if self.launch is not None and (self.workspace or self.ignore):
            raise ValueError(
                "workspace and ignore describe declared launches; a launch "
                "plan builds its own"
            )
        return self

    @property
    def declared_steps(self) -> tuple[CudaStep, ...]:
        if self.kernel is not None:
            return (CudaKernel(kernel=self.kernel),)
        return self.steps

    @property
    def name(self) -> str:
        """Route name: the plan's name, the kernel's unqualified name, or the
        KernelSpec name for declared steps."""
        if self.launch is not None:
            return self.launch.__name__
        if self.kernel is not None:
            return self.declared_steps[0].name
        return self.spec.name

    @property
    def _key(self) -> tuple[str, str]:
        return self.extension, self.name


@dataclass(frozen=True, slots=True)
class _CompiledCudaRoute:
    """Complete construction-time launch ABI consumed by trusted dispatch."""

    extension: str
    name: str
    launch: LaunchPlan | None
    steps: tuple | None
    workspace: Mapping[str, CudaWorkspace]
    check: Callable | None
    check_args: tuple[str, ...]
    spec: KernelSpec
    launch_args: tuple[str, ...]
    projection: CudaNativeProjection
    omitted: frozenset[str]
    resolved_specs: tuple[KernelSpec, ...]


def _validate_projection_values(
    spec: KernelSpec,
    projection: CudaNativeProjection,
    omitted: set[str],
) -> None:
    """Validate every omitted canonical value against its declared ABI kind."""

    for name in sorted(omitted):
        value = projection.fixed[name]
        if name in spec.buffers:
            if name not in spec.optional_buffers:
                raise ValueError(
                    f"{spec.name}: CUDA launch plan omits required canonical "
                    f"buffer {name!r}"
                )
            if value is not None:
                raise ValueError(
                    f"{spec.name}: omitted optional CUDA buffer {name!r} "
                    "must be fixed to None"
                )
            continue

        kind = spec.runtime_scalars.get(name, spec.compile_time.get(name))
        validation_kind = "float32" if kind == "precision" else kind
        if validation_kind is None or not _host_scalar_is_valid(
            value,
            validation_kind,
        ):
            raise ValueError(
                f"{spec.name}: CUDA native projection value {name!r} must "
                f"be an exact finite {kind} host scalar, got {value!r} "
                f"({type(value).__name__})"
            )


def _plan_parameters(spec: KernelSpec, plan: LaunchPlan) -> tuple[str, ...]:
    """Canonical names a plan receives; ``**values`` receives all of them."""

    parameters = inspect.signature(plan).parameters.values()
    unsupported = [
        parameter.name
        for parameter in parameters
        if parameter.kind
        not in (
            parameter.KEYWORD_ONLY,
            parameter.POSITIONAL_OR_KEYWORD,
            parameter.VAR_KEYWORD,
        )
        or parameter.default is not parameter.empty
    ]
    if unsupported:
        raise ValueError(
            f"{spec.name}: CUDA launch plan {plan.__name__} parameters must be "
            f"plain keyword parameters without defaults: {unsupported}"
        )
    names = tuple(
        parameter.name
        for parameter in parameters
        if parameter.kind is not parameter.VAR_KEYWORD
    )
    if any(parameter.kind is parameter.VAR_KEYWORD for parameter in parameters):
        names += tuple(
            name for name in (*spec.parameters, "BLOCK_SIZE") if name not in names
        )
    return names


def _compile_cuda_route(route: CudaRoute) -> _CompiledCudaRoute:
    """Validate one route's canonical ABI exactly once during construction."""

    spec = route.spec
    projection = route.projection or CudaNativeProjection()
    # Mask members are supplied by source specialization, not plan inputs.
    grouped_features = (
        set().union(*spec.compile_time_masks.values())
        if spec.compile_time_masks
        else set()
    )
    ignored = set(route.ignore)
    canonical = set(spec.parameters)
    unknown_ignored = ignored.difference(canonical)
    if unknown_ignored:
        raise ValueError(
            f"{spec.name}: CUDA route ignores non-canonical values "
            f"{sorted(unknown_ignored)}"
        )
    steps = None
    if route.launch is None:
        steps = compile_steps(
            spec,
            route.declared_steps,
            route.workspace,
            grouped_features | ignored | set(projection.fixed),
        )
        launch_args = (
            *(
                name
                for name in spec.parameters
                if name not in ignored and name not in projection.fixed
            ),
            "BLOCK_SIZE",
        )
    else:
        launch_args = _plan_parameters(spec, route.launch)
    check_args = () if route.check is None else _plan_parameters(spec, route.check)
    names = set(launch_args)
    if "BLOCK_SIZE" not in names:
        raise ValueError(
            f"{spec.name}: CUDA launch plan must take compiler-owned BLOCK_SIZE"
        )
    unknown = names.union(check_args).difference(canonical, {"BLOCK_SIZE"})
    if unknown:
        raise ValueError(
            f"CUDA launch plan arguments are outside canonical ABI: {sorted(unknown)}"
        )
    omitted = canonical.difference(names)
    unknown_fixed = set(projection.fixed).difference(omitted)
    if unknown_fixed:
        raise ValueError(
            f"{spec.name}: CUDA native projection fixes values that are still "
            "consumed by the launch plan or absent from KernelSpec: "
            f"{sorted(unknown_fixed)}"
        )
    missing_fixed = omitted.difference(projection.fixed, grouped_features, ignored)
    if missing_fixed:
        raise ValueError(
            f"{spec.name}: CUDA launch plan omits canonical inputs "
            f"{sorted(missing_fixed)}; define every omitted value in "
            "CudaNativeProjection.fixed instead of inferring semantics from "
            "an absent plan parameter"
        )
    _validate_projection_values(
        spec, projection, omitted.difference(grouped_features, ignored)
    )
    return _CompiledCudaRoute(
        extension=route.extension,
        name=route.name,
        launch=route.launch,
        steps=steps,
        workspace=_immutable_dict(dict(route.workspace)),
        check=route.check,
        check_args=check_args,
        spec=spec,
        launch_args=launch_args,
        projection=projection,
        omitted=frozenset(omitted),
        resolved_specs=(
            (
                spec._resolve_precision("float32"),
                spec._resolve_precision("float64"),
            )
            if spec._uses_precision
            else (spec,)
        ),
    )


_CUDA_FACTORY_CONTEXT = "hydroforge_cuda_extension_group"


class _CudaFactoryRequest(HydroForgeModel):
    extension: str
    launch: str

    _route: _CompiledCudaRoute = PrivateAttr()

    @model_validator(mode="after")
    def _resolve_route(self, info: ValidationInfo):
        group = (
            info.context.get(_CUDA_FACTORY_CONTEXT)
            if isinstance(info.context, Mapping)
            else None
        )
        if group is None:
            raise ValueError("CUDA factory request requires group context")
        try:
            self._route = group._route_index[(self.extension, self.launch)]
        except KeyError as error:
            raise ValueError(
                f"unknown CUDA route {self.extension!r}/{self.launch!r}"
            ) from error
        return self

    @property
    def route(self) -> _CompiledCudaRoute:
        return self._route


class _CudaDispatcherDeclaration(HydroForgeModel):
    """Bind a validated registry precision to one compiled route."""

    route: _CompiledCudaRoute
    spec: KernelSpec

    @model_validator(mode="after")
    def _validate_spec(self):
        if self.spec not in self.route.resolved_specs:
            raise ValueError(
                f"CUDA route {self.route.extension!r}/"
                f"{self.route.name!r} declares KernelSpec "
                f"{self.route.spec.name!r}, not {self.spec.name!r}"
            )
        return self


class CudaExtensionGroup(HydroForgeModel):
    """A named namespace of device sources and their launch plans."""

    specs: FrozenMapping[Identifier, CudaExtensionSpec] = Field(min_length=1)
    routes: tuple[CudaRoute, ...] = Field(min_length=1)

    _route_index: Mapping[tuple[str, str], _CompiledCudaRoute] = PrivateAttr(
        default_factory=dict,
    )
    _programs: Mapping[str, rtc.RtcProgram] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_group(self) -> Self:
        routes: dict[tuple[str, str], _CompiledCudaRoute] = {}
        for route in self.routes:
            if route.extension not in self.specs:
                raise ValueError(
                    f"CUDA route {route.extension!r}/{route.name!r} "
                    "references an unknown extension"
                )
            if route._key in routes:
                raise ValueError(
                    f"CUDA route {route.extension!r}/{route.name!r} "
                    "is declared more than once"
                )
            try:
                routes[route._key] = _compile_cuda_route(route)
            except (TypeError, ValueError, OverflowError) as error:
                raise ValueError(str(error)) from error
        unused = sorted(set(self.specs).difference(key[0] for key in routes))
        if unused:
            raise ValueError(
                f"CUDA extension specs must each declare at least one route: {unused}"
            )
        self._route_index = _immutable_dict(routes)
        self._programs = _immutable_dict(
            {name: spec.program(name) for name, spec in self.specs.items()}
        )
        return self

    def factory(
        self,
        extension: str,
        launch: str,
    ) -> Callable[[], CudaDispatcher]:
        """Return the registry factory for one route, named as ``CudaRoute.name``."""

        request = _CudaFactoryRequest.model_validate(
            {"extension": extension, "launch": launch},
            context={_CUDA_FACTORY_CONTEXT: self},
        )
        route = request.route
        program = self._programs[extension]

        @registry_factory
        def factory() -> CudaDispatcher:
            declaration = _CudaDispatcherDeclaration(
                route=route,
                spec=active_kernel_spec(),
            )
            return CudaDispatcher(route, program, spec=declaration.spec)

        return factory


def _launch_device(values: Mapping[str, Any]) -> int:
    for value in values.values():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            return value.device.index
    return torch.cuda.current_device()


class CudaDispatcher:
    """Trusted adapter from canonical values to one runtime-compiled launch."""

    def __init__(
        self,
        route: _CompiledCudaRoute,
        program: rtc.RtcProgram,
        *,
        spec: KernelSpec,
    ) -> None:
        self.route = route
        self.program = program
        self.parameters = spec.parameters
        self.launch_args = route.launch_args
        self.projection = route.projection
        self.omitted = route.omitted
        self.spec = spec
        self.__hydroforge_kernel__ = spec._canonical_metadata
        self.__hydroforge_lowering__ = BackendLoweringSpec.canonical(
            buffer_elements="tensor",
        )

    def _validate_specialization_input(
        self,
        values: Mapping[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI,
    ) -> None:
        """Backend-specific validation invoked only by the Pydantic request."""

        del buffer_dtypes
        block_size = values["BLOCK_SIZE"]
        if not 1 <= block_size <= 1024:
            raise ValueError(
                f"{self.spec.name}: CUDA BLOCK_SIZE must be an exact int in "
                f"[1, 1024], got {block_size!r}"
            )
        self.projection._validate(values, kernel=self.spec.name)
        if self.route.check is not None:
            self.route.check(**{name: values[name] for name in self.route.check_args})

    def _is_empty(self, values: Mapping[str, Any]) -> bool:
        size_keys = (
            (self.spec.size_key,)
            if isinstance(self.spec.size_key, str)
            else self.spec.size_key
        )
        extent = 1
        for name in size_keys:
            extent *= values[name]
        return extent == 0

    def _program_for(self, values: Mapping[str, Any], entries: str) -> rtc.RtcProgram:
        """Specialize grouped masks as definitions and append generated entries."""

        if not self.spec.compile_time_masks and not entries:
            return self.program
        prefix = "".join(
            f"#define HYDROFORGE_{name} {self.spec.compile_time_mask(name, values)}u\n"
            for name in self.spec.compile_time_masks
        )
        return rtc.RtcProgram(
            prefix + self.program.source + entries,
            self.program.options,
            self.program.name,
        )

    def _plan(
        self, values: Mapping[str, Any], buffer_dtypes: BufferDTypeABI | None
    ) -> tuple[rtc.RtcRequest | None, tuple, int, dict[str, torch.Tensor]]:
        device = _launch_device(values)
        workspace = {}
        entries = ""
        if self.route.steps is not None:
            workspace = allocate_workspace(self.route.workspace, values, device)
            steps, entries = render_steps(
                self.spec, self.route.steps, values, workspace, buffer_dtypes
            )
        else:
            steps = tuple(
                self.route.launch(**{name: values[name] for name in self.launch_args})
            )
        request = (
            rtc.request_for(self._program_for(values, entries), steps)
            if any(isinstance(step, rtc.CudaLaunch) for step in steps)
            else None
        )
        return request, steps, device, workspace

    def _precompile_arguments(
        self,
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI | None = None,
    ) -> tuple[rtc.RtcRequest, int] | None:
        if self._is_empty(arguments):
            return None
        request, _steps, device, _workspace = self._plan(arguments, buffer_dtypes)
        return None if request is None else (request, device)

    def specialize(
        self,
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI,
    ) -> Callable[[], None]:
        if self._is_empty(arguments):

            def no_op() -> None:
                return None

            return no_op
        request, steps, device, workspace = self._plan(arguments, buffer_dtypes)
        if request is None:

            def run() -> None:
                for step in steps:
                    step()

            return run
        launch = rtc.prepare(request, steps, device)
        # The prepared arguments hold raw addresses; keep the scratch alive.
        launch.workspace = workspace
        return launch


__all__ = [
    "CudaDispatcher",
    "CudaExtensionGroup",
    "CudaNativeProjection",
    "CudaRoute",
]
