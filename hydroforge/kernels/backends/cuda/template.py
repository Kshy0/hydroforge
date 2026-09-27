"""KernelSpec-first runtime-compiled CUDA kernels."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import Field, PrivateAttr, model_validator

from hydroforge.contracts.kernels import (
    BackendLoweringSpec,
    BufferDTypeABI,
    KernelSpec,
)
from hydroforge.contracts.naming import Identifier
from hydroforge.contracts.validation import FrozenMapping, HydroForgeModel
from hydroforge.kernels.backends.cuda import rtc
from hydroforge.kernels.backends.cuda.dispatcher import (
    CudaDispatcher,
    CudaNativeProjection,
    CudaRoute,
    LaunchPlan,
    _compile_cuda_route,
    _plan_parameters,
)
from hydroforge.kernels.backends.cuda.launch import (
    CudaStep,
    CudaWorkspace,
    constant_literal,
)
from hydroforge.kernels.backends.cuda.spec import (
    CudaExtensionSpec,
    cuda_physics_options,
)
from hydroforge.kernels.context import resolve_factory_spec

_SCALAR_TYPES = {
    "bool": "bool",
    "int32": "int",
    "uint32": "uint32_t",
    "index": "int64_t",
    "float32": "float",
    "float64": "double",
}


def cuda_compile_time_source(spec: KernelSpec, constants: dict[str, Any]) -> str:
    """Declare canonical constants and their zero-storage C++ parameter type."""
    expected = set(spec.compile_time)
    supplied = set(constants)
    if supplied != expected:
        raise TypeError(
            f"{spec.name}: CUDA source specialization requires exact compile-time "
            f"values; missing={sorted(expected - supplied)}, "
            f"extra={sorted(supplied - expected)}"
        )
    spec._validate_compile_time(constants)
    declarations = [
        f"static constexpr {_SCALAR_TYPES[kind]} {name} = "
        f"{constant_literal(kind, constants[name])};"
        for name, kind in spec.compile_time.items()
    ]
    declarations.extend(
        f"static constexpr uint32_t {name} = "
        f"{spec.compile_time_mask(name, constants)}u;"
        for name in spec.compile_time_masks
    )
    members = "\n".join(
        f"    static constexpr auto {name} = ::{name};"
        for name in (*spec.compile_time, *spec.compile_time_masks)
    )
    return "\n".join(declarations) + (
        f"\nnamespace hydroforge {{\nstruct KernelParameters {{\n{members}\n}};\n}}\n"
    )


class SpecCudaTemplateDispatcher:
    """One device source specialized by a canonical KernelSpec's constants.

    The source holds device code only. Each distinct compile-time tuple renders
    the constants ahead of it. ``entry`` holds the :class:`CudaRoute` fields
    that map canonical values to launches: a ``kernel`` declaration or a
    ``launch`` plan. Compile-time values are defined by the source, so they are
    never kernel arguments.
    """

    def __init__(
        self,
        spec: KernelSpec,
        source: str,
        *,
        entry: dict[str, Any],
        options: tuple[str, ...] = (),
    ) -> None:
        if not isinstance(spec, KernelSpec):
            raise TypeError("CUDA template requires a KernelSpec")
        if type(source) is not str or not source.strip():
            raise ValueError("CUDA template source must be a non-empty string")
        if type(options) is not tuple or any(
            type(option) is not str or not option for option in options
        ):
            raise TypeError("CUDA template options must be a string tuple")
        self.spec = spec
        self.source = source
        self.entry = entry
        self.options = options
        launch = entry.get("launch")
        self._plan_parameters = (
            set() if launch is None else set(_plan_parameters(spec, launch))
        )
        self._dispatchers: dict[tuple[Any, ...], CudaDispatcher] = {}
        self.__hydroforge_kernel__ = spec._canonical_metadata
        self.__hydroforge_lowering__ = BackendLoweringSpec.canonical(
            buffer_elements="tensor",
        )

    def source_for(self, compile_time: dict[str, Any] | None = None) -> str:
        """Return the deterministic generated device source for cold-path audit."""

        constants = {} if compile_time is None else compile_time
        return cuda_compile_time_source(self.spec, constants) + self.source

    def _specialization_key(self, arguments: dict[str, Any]) -> tuple[str, ...]:
        return tuple(
            constant_literal(kind, arguments[name])
            for name, kind in self.spec.compile_time.items()
        )

    def _dispatcher_for(self, arguments: dict[str, Any]) -> CudaDispatcher:
        key = self._specialization_key(arguments)
        dispatcher = self._dispatchers.get(key)
        if dispatcher is None:
            constants = {name: arguments[name] for name in self.spec.compile_time}
            route = _compile_cuda_route(
                CudaRoute(
                    extension="template",
                    **self.entry,
                    spec=self.spec,
                    projection=CudaNativeProjection(
                        fixed={
                            name: value
                            for name, value in constants.items()
                            if name not in self._plan_parameters
                        }
                    ),
                )
            )
            program = rtc.RtcProgram(
                self.source_for(constants),
                cuda_physics_options(self.options),
                self.spec.name,
            )
            dispatcher = CudaDispatcher(route, program, spec=self.spec)
            self._dispatchers[key] = dispatcher
        return dispatcher

    def _validate_specialization_input(
        self,
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI,
    ) -> None:
        """Build and validate the concrete source ABI inside Pydantic."""

        self._dispatcher_for(arguments)._validate_specialization_input(
            arguments,
            buffer_dtypes=buffer_dtypes,
        )

    def specialize(
        self,
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI,
    ):
        dispatcher = self._dispatchers[self._specialization_key(arguments)]
        return dispatcher.specialize(arguments, buffer_dtypes=buffer_dtypes)

    def _precompile_arguments(
        self,
        arguments: dict[str, Any],
        *,
        buffer_dtypes: BufferDTypeABI | None = None,
    ):
        dispatcher = self._dispatchers[self._specialization_key(arguments)]
        return dispatcher._precompile_arguments(arguments, buffer_dtypes=buffer_dtypes)


class _SpecCudaDispatcherDeclaration(HydroForgeModel):
    spec: KernelSpec | None = None
    source: str | Annotated[Path, Field(strict=False)]
    launch: Any = None
    kernel: str | None = None
    steps: tuple[CudaStep, ...] = ()
    workspace: FrozenMapping[Identifier, CudaWorkspace] = Field(default_factory=dict)
    ignore: tuple[Identifier, ...] = ()
    check: Any = None
    include_root: Path | None = Field(default=None, strict=False)
    options: tuple[str, ...] = ()

    _dispatcher: SpecCudaTemplateDispatcher = PrivateAttr()

    @model_validator(mode="after")
    def _build(self):
        try:
            spec = resolve_factory_spec(self.spec, factory="make_spec_cuda_dispatcher")
            source = self.source
            if isinstance(source, Path):
                source = CudaExtensionSpec(
                    source=source,
                    include_root=self.include_root,
                )._materialize_source()
            elif self.include_root is not None:
                raise ValueError("CUDA include_root requires a source Path")
            entry = {
                "launch": self.launch,
                "kernel": self.kernel,
                "steps": self.steps,
                "workspace": dict(self.workspace),
                "ignore": self.ignore,
                "check": self.check,
            }
            # Validate the entry once up front; specializations reuse it.
            CudaRoute(extension="template", spec=spec, **entry)
            self._dispatcher = SpecCudaTemplateDispatcher(
                spec,
                source,
                entry=entry,
                options=self.options,
            )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(str(error)) from error
        return self


def make_spec_cuda_dispatcher(
    spec: KernelSpec | None = None,
    *,
    source: str | Path,
    launch: LaunchPlan | None = None,
    kernel: str | None = None,
    steps: tuple[CudaStep, ...] = (),
    workspace: dict[str, CudaWorkspace] | None = None,
    ignore: tuple[str, ...] = (),
    check: Any = None,
    include_root: Path | None = None,
    options: tuple[str, ...] = (),
) -> SpecCudaTemplateDispatcher:
    """Create a lazy runtime-compiled implementation from the active Spec.

    ``kernel``, ``steps``, ``workspace``, ``ignore``, ``check`` and ``launch``
    mean what they mean for :class:`CudaRoute`; compile-time values are
    defined by the source, so they are never kernel arguments.
    """

    return _SpecCudaDispatcherDeclaration(
        spec=spec,
        source=source,
        launch=launch,
        kernel=kernel,
        steps=steps,
        workspace=workspace or {},
        ignore=ignore,
        check=check,
        include_root=include_root,
        options=options,
    )._dispatcher


__all__ = [
    "cuda_compile_time_source",
    "make_spec_cuda_dispatcher",
]
